import os
import re
import sys
from functools import partial
from typing import *

import torch

import qelos as q
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from nltk import PorterStemmer

from torch.utils.data import DataLoader

from funcparse.decoding import TransitionModel, TFActionSeqDecoder, LSTMCellTransition, BeamActionSeqDecoder
from funcparse.grammar import FuncGrammar, passtr_to_pas
from funcparse.states import FuncTreeState, FuncTreeStateBatch
from funcparse.vocab import VocabBuilder, SentenceEncoder, FuncQueryEncoder
from funcparse.nn import TokenEmb, PtrGenOutput


def build_type_grammar_from(outputs:List[str], inputs:List[str], tokenizer):
    # get all predicates, terminals and words from queries
    # example inputs: and(wife(president(US), BO), wife(president(countryid('united states')))
    g = FuncGrammar("<R>")
    # DONETODO: IMPORTANT (BREAKING): can not allow rules of the forms <R> -> <W>* <W>* or <R> -> <W>* <R>, rules may only contain a single argument if it is of sibling type (*)
    g.add_rule("<R> -> cityid :: <CITYNAME> <CITYSTATE>")
    g.add_rule("<CITYNAME> -> cityname :: <W>*")
    g.add_rule("<CITYSTATE> -> citystate :: <W>*")
    def _rec_process_pas(x):
        if isinstance(x, tuple):    # parent
            if x[0] == "cityid":
                cityname = ("cityname", [x[1][0]])
                citystate = ("citystate", [x[1][1] if x[1][1] != "_" else f"'_'"])
                x = (x[0], [cityname, citystate])
            b = [_rec_process_pas(a) for a in x[1]]
            child_types, subtrees = zip(*b)
            rule_exists = False
            if x[0] in g.rules_by_arg:
                existing_rules = g.rules_by_arg[x[0]]
                existing_rules_ = []
                for rule in existing_rules:
                    rettype, rulebody = rule.split(" -> ")
                    if "::" not in rulebody:
                        continue
                    rule_arg, rule_children = rulebody.split(" :: ")
                    if child_types == tuple(rule_children.split()):
                        existing_rules_.append(rule)
                if len(existing_rules_) == 1:
                    rule_exists = True
                    rettype, rulebody = existing_rules_[0].split(" -> ")
            if not rule_exists:
                rettype = "<R>"
                rule = f"{rettype} -> {x[0]} :: {' '.join(child_types)}"
                g.add_rule(rule)
            treestr = f"{x[0]}({', '.join([', '.join(subtree) for subtree in subtrees])})"
            return rettype, [treestr]
        elif x[0] == "'" and x[-1] == "'":   # string
            tokens = tokenizer(x[1:-1])
            for token in tokens:
                rule = f"<W>* -> '{token}' -- <W>*"
                g.add_rule(rule)
            return "<W>*", [f"'{token}'" for token in tokens] + ["@W:END@"]
        # elif x == "_":  # empty string
        #     return _rec_process_pas("'_'")
        else:   # terminal
            rule = f"<R> -> {x}"
            g.add_rule(rule)
            return "<R>", [x]

    pre_parsed_queries = []
    for query in outputs:
        pas = passtr_to_pas(query)
        _, pre_parsed_query = _rec_process_pas(pas)
        pre_parsed_queries.append(pre_parsed_query[0])

    g.add_rule("<W>* -> @W:END@")

    # get all words from questions
    for question in inputs:
        tokens = tokenizer(question)
        for token in tokens:
            rule = f"<W>* -> '{token}' -- <W>*"
            g.add_rule(rule)
    return g, pre_parsed_queries


def try_build_grammar(
                 geoquery_path:str="../../data/geoquery",
                 questions_file:str="questions.txt",
                 queries_file:str="queries.funql",):
    tt = q.ticktock("try_grammar")
    tt.tick("building grammar")
    inputs = [x.strip() for x in open(os.path.join(geoquery_path, questions_file), "r").readlines()]
    outputs = [x.strip() for x in open(os.path.join(geoquery_path, queries_file), "r").readlines()]
    tokenizer = lambda x: x.split()
    g, preparsed = build_type_grammar_from(outputs, inputs, tokenizer)
    tt.tock("grammar built")
    tt.tick("parsing")
    print(g)
    print(g.rules_by_type)
    N = len(outputs)
    for before, after in zip(outputs[:N], preparsed[:N]):
        print(before)
        print(after)
        print(g.actions_for(after, format="prolog"))
        print(g.actions_to_tree(g.actions_for(after, format="prolog")))
    tt.tock("parsed")
    print()
    print(g.rules_by_arg["stateid"])
    print(g.rules_by_arg["cityid"])



class GeoQueryDataset(object):
    def __init__(self,
                 geoquery_path:str="../../data/geoquery",
                 questions_file:str="questions.txt",
                 queries_file:str="queries.funql",
                 train_indexes:str="train_indexes.txt",
                 test_indexes:str="test_indexes.txt",
                 sentence_encoder:SentenceEncoder=None,
                 min_freq:int=2,
                 **kw):
        super(GeoQueryDataset, self).__init__(**kw)
        self.data = {}

        self.sentence_encoder = sentence_encoder

        inputs = [x.strip() for x in open(os.path.join(geoquery_path, questions_file), "r").readlines()]
        outputs = [x.strip() for x in open(os.path.join(geoquery_path, queries_file), "r").readlines()]
        t_idxs = set([int(x.strip()) for x in open(os.path.join(geoquery_path, train_indexes), "r").readlines()])
        x_idxs = set([int(x.strip()) for x in open(os.path.join(geoquery_path, test_indexes), "r").readlines()])


        # build input vocabulary
        for i, inp in enumerate(inputs):
            self.sentence_encoder.inc_build_vocab(inp, seen=i in t_idxs)
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)

        self.grammar, preparsed_queries = build_type_grammar_from(outputs, inputs, sentence_encoder.tokenizer)

        self.query_encoder = FuncQueryEncoder(self.grammar, sentence_encoder=self.sentence_encoder, format="prolog")


        splits = [None] * len(inputs)
        for t_idx in t_idxs:
            splits[t_idx] = "train"
        for x_idx in x_idxs:
            splits[x_idx] = "test"
        assert(all([split != None for split in splits]))

        for i, out in enumerate(preparsed_queries):
            self.query_encoder.inc_build_vocab(out, seen=i in t_idxs)
        self.query_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(inputs, preparsed_queries, splits)

    def build_data(self, inputs:Iterable[str], outputs:Iterable[str], splits:Iterable[str]):
        for inp, out, split in zip(inputs, outputs, splits):
            gs = FuncTreeState(inp, out, self.sentence_encoder, self.query_encoder)
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(gs)

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    @staticmethod
    def collate_fn(data:Iterable):
        ret = FuncTreeStateBatch(data)
        return ret


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = GeoQueryDataset(sentence_encoder=SentenceEncoder(tokenizer=lambda x: x.split()))
    tt.tock("dataset built")


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_dataloaders(ds:GeoQueryDataset, batsize:int=5):
    dls = {}
    for split in ds.data.keys():
        dls[split] = DataLoader(ds.get_split(split), batch_size=batsize, shuffle=split=="train",    # put shuffle back
             collate_fn=GeoQueryDataset.collate_fn)
    return dls


class BasicPtrGenModel(TransitionModel):
    def __init__(self, inp_emb, inp_enc, out_emb, out_rnn, out_lin, att, dropout=0., **kw):
        super(BasicPtrGenModel, self).__init__(**kw)
        self.inp_emb, self.inp_enc = inp_emb, inp_enc
        self.out_emb, self.out_rnn, self.out_lin = out_emb, out_rnn, out_lin
        self.att = att
        # self.ce = q.CELoss(reduction="none", ignore_index=0, mode="probs")
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x:FuncTreeStateBatch):
        if "ctx" not in x.batched_states:
            # encode input
            inptensor = x.batched_states["inp_tensor"]
            mask = inptensor != 0
            inpembs = self.inp_emb(inptensor)
            inpembs = self.dropout(inpembs)
            inpenc = self.inp_enc(inpembs, mask)
            x.batched_states["ctx"] = inpenc
            x.batched_states["ctx_mask"] = mask

        ctx = x.batched_states["ctx"]
        ctx_mask = x.batched_states["ctx_mask"]

        emb = self.out_emb(x.batched_states["prev_action"])

        if "rnn" not in x.batched_states:
            x.batched_states["rnn"] = {}
        enc = self.out_rnn(emb, x.batched_states["rnn"])

        alphas, summ, scores = self.att(enc, ctx, ctx_mask)
        enc = torch.cat([enc, summ], -1)

        probs, ptr_or_gen, gen_probs, ptr_position_probs = self.out_lin(enc, x, alphas)
        return probs, x

        _, rules = probs.max(-1)
        _, actions_ptr_or_gen = ptr_or_gen.max(-1)
        _, ptr_positions = ptr_position_probs.max(-1)
        _, actions_gen = gen_probs.max(-1)


        # predicted actions
        predicted_actions = []
        for i, (state, ptr_or_gen_e, gen_action_e, copy_action_e) \
                in enumerate(zip(x.states,
                                 list(actions_ptr_or_gen.cpu()),
                                 list(actions_gen.cpu()),
                                 list(ptr_positions.cpu()))):
            if not state.is_terminated:
                open_node = state.open_nodes[0]
                if ptr_or_gen_e.item() == 0:  # gen
                    action = state.query_encoder.vocab_actions(gen_action_e.item())
                    _rule = action
                else:
                    action = f"COPY[{copy_action_e}]"
                    _rule = f"<W>* -> '{state.inp_tokens[copy_action_e]}' -- <W>*"
                predicted_actions.append(action)
                state.pred_actions.append(action)
                state.pred_rules.append(_rule)
            else:
                predicted_actions.append(state.query_encoder.none_action)

        if x.states[0].has_gold:    # compute loss and accuracies
            gold_actions = torch.zeros_like(actions)
            term_mask = torch.zeros_like(actions).float()
            for i, state in enumerate(x.states):
                if not state.is_terminated:
                    open_node = state.open_nodes[0]
                    gold_actions[i] = state.query_encoder.vocab_actions[state.get_gold_action_at(open_node)]
                    term_mask[i] = 1
                else:
                    term_mask[i] = 0
            loss = self.ce(probs, gold_actions)
            acc = gold_actions == actions
            ret = (loss, acc, term_mask)

            x.batched_states["prev_action"] = gold_actions

            # advance states
            for i, state in enumerate(x.states):
                if not state.is_terminated:
                    open_node = state.open_nodes[0]
                    gold_action = state.get_gold_action_at(open_node)
                    state.apply_action(open_node, gold_action)
            return x, ret
        else:
            x.batched_states["prev_action"] = actions
            for action, state in zip(predicted_actions, x.states):
                if not state.is_terminated:
                    state.apply_action(open_node, action)
                else:
                    assert(action == state.query_encoder.none_action)
            return x


def create_model(embdim=100, hdim=100, dropout=0., numlayers:int=1,
                 sentence_encoder:SentenceEncoder=None, query_encoder:FuncQueryEncoder=None):
    inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
    inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(embdim, hdim, num_layers=numlayers, bidirectional=True, batch_first=True,
                      dropout=dropout))
    decoder_emb = torch.nn.Embedding(query_encoder.vocab_tokens.number_of_ids(), embdim, padding_idx=0)
    decoder_emb = TokenEmb(decoder_emb, rare_token_ids=query_encoder.vocab_tokens.rare_ids, rare_id=1)
    decoder_rnn = [torch.nn.LSTMCell(embdim, hdim * 2)]
    for i in range(numlayers - 1):
        decoder_rnn.append(torch.nn.LSTMCell(hdim * 2, hdim * 2))
    decoder_rnn = LSTMCellTransition(*decoder_rnn, dropout=dropout)
    decoder_out = PtrGenOutput(hdim*4, sentence_encoder, query_encoder)
    attention = q.Attention(q.MatMulDotAttComp(hdim*2, hdim*2))
    model = BasicPtrGenModel(inpemb, encoder, decoder_emb, decoder_rnn, decoder_out, attention)
    dec = TFActionSeqDecoder(model)
    return dec


def run(lr=0.001,
        batsize=50,
        epochs=30,
        embdim=100,
        numlayers=2,
        dropout=.2,
        wreg=1e-6,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        beamsize=5,
        fulltest=False,
        ):
    # DONE: Porter stemmer
    # DONE: linear attention
    # DONE: grad norm
    # TODO: beam search
    # TODO: different optimizer?
    # TODO: lr scheduler
    tt = q.ticktock("script")
    ttt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    stemmer = PorterStemmer()
    tokenizer = lambda x: [stemmer.stem(xe) for xe in x.split()]
    ds = GeoQueryDataset(sentence_encoder=SentenceEncoder(tokenizer=tokenizer), min_freq=minfreq)
    dls = get_dataloaders(ds, batsize=batsize)
    train_dl = dls["train"]
    test_dl = dls["test"]
    tt.tock("data loaded")

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    tfdecoder = create_model(embdim=embdim, hdim=embdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder)
    beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize)

    # # test
    # tt.tick("doing one epoch")
    # for batch in iter(train_dl):
    #     batch = batch.to(device)
    #     ttt.tick("start batch")
    #     # with torch.no_grad():
    #     out = tfdecoder(batch)
    #     ttt.tock("end batch")
    # tt.tock("done one epoch")
    # print(out)
    # sys.exit()

    # beamdecoder(next(iter(train_dl)))

    # print(dict(tfdecoder.named_parameters()).keys())

    losses = [q.LossWrapper(q.SelectedLinearLoss(x, reduction=None), name=x) for x in ["loss", "any_acc", "seq_acc"]]
    vlosses = [q.LossWrapper(q.SelectedLinearLoss(x, reduction=None), name=x) for x in ["seq_acc", "tree_acc"]]

    # 4. define optim
    optim = torch.optim.Adam(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # 6. define training function (using partial)
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=train_dl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=beamdecoder, dataloader=test_dl, losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")



if __name__ == '__main__':
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)