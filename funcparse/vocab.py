from abc import ABC, abstractmethod
from typing import Union, Callable, List

import torch

from funcparse.grammar import FuncGrammar


class Vocab(object):
    padtoken = "@PAD@"
    unktoken = "@UNK@"
    def __init__(self, padid:int=0, unkid:int=1, **kw):
        self.D = {self.padtoken: padid, self.unktoken: unkid}
        self._RD = {v: k for k, v in self.D.items()}
        self.growing = True

    @property
    def RD(self):
        if self.growing is True:
            self._RD = {v: k for k, v in self.D.items()}
        else:
            return self._RD

    def nextid(self):
        return max(self.D.values()) + 1

    def stopgrowth(self):
        self.RD
        self.growing = False

    def __getitem__(self, item:str) -> int:
        if item not in self.D:
            if self.growing:
                id = self.nextid()
                self.D[item] = id
            else:
                assert(self.unktoken in self.D)
                item = self.unktoken
        id = self.D[item]
        return id

    def __call__(self, item:int) -> str:
        return self.RD[item]

    def number_of_ids(self):
        return max(self.D.values()) + 1

    def reverse(self):
        return {v: k for k, v in self.D.items()}

    def __iter__(self):
        return iter([(k, v) for k, v in self.D.items()])

    def __contains__(self, item:Union[str,int]):
        if isinstance(item, str):
            return item in self.D
        if isinstance(item, int):
            return item in self.RD
        else:
            raise Exception("illegal argument")


class VocabBuilder(ABC):
    @abstractmethod
    def inc_build_vocab(self, x:str):
        raise NotImplemented()

    @abstractmethod
    def finalize_vocab(self):
        raise NotImplemented()

    @abstractmethod
    def vocabs_finalized(self):
        raise NotImplemented()

    
class SentenceEncoder(VocabBuilder):
    def __init__(self, tokenizer:Callable[[str], List[str]], vocab:Vocab=None, **kw):
        super(SentenceEncoder, self).__init__(**kw)
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else Vocab()
        self.vocab_final = False
        
    def inc_build_vocab(self, x:str):
        if not self.vocab_final:
            tokens = self.tokenizer(x)
            ids = [self.vocab[token] for token in tokens]
    
    def finalize_vocab(self):
        self.vocab_final = True
        self.vocab.stopgrowth()
        
    def vocabs_finalized(self):
        return self.vocab_final
    
    def convert(self, x:str, return_what="tensor"):     # "tensor", "ids", "tokens" or comma-separated combo of all
        rets = [r.strip() for r in return_what.split(",")]
        tokens = self.tokenizer(x)
        ids = [self.vocab[token] for token in tokens]
        tensor = torch.tensor(ids, dtype=torch.long)
        ret = {"tokens": tokens, "ids": ids, "tensor": tensor}
        ret = [ret[r] for r in rets]
        return ret
    
    
class FuncQueryEncoder(VocabBuilder):
    string_types = ("<W>+", "<W>*")
    def __init__(self, grammar:FuncGrammar=None, vocab_tokens:Vocab=None, vocab_actions:Vocab=None,
                 sentence_encoder:SentenceEncoder=None, format:str="prolog", **kw):
        super(FuncQueryEncoder, self).__init__(**kw)
        self.vocab_final = False
        self.vocab_tokens = vocab_tokens if vocab_tokens is not None else Vocab()
        self.vocab_actions = vocab_actions if vocab_actions is not None else Vocab()

        self.none_action = "[NONE]"
        self.start_action = "@START@"
        self.vocab_actions[self.none_action]
        self.vocab_actions[self.start_action]

        self.grammar = grammar
        self.sentence_encoder = sentence_encoder

        self.format = format

        self.string_actions = set()

    def vocabs_finalized(self):
        return self.vocab_final

    def inc_build_vocab(self, x:str):
        pass

    def finalize_vocab(self):
        for out_type, rules in self.grammar.rules_by_type.items():
            for rule in rules:
                self.vocab_actions[rule]
                head, body = rule.split(" -> ")
                if head in self.string_types:
                    self.string_actions.add(rule)

                self.vocab_tokens[head]
                body = body.split(" ")
                body = [x for x in body if x not in ["::", "--"]]
                for x in body:
                    self.vocab_tokens[x]

        self.vocab_tokens.stopgrowth()
        self.vocab_actions.stopgrowth()

        self.vocab_final = True

    def grammar_actions_for(self, x:str):
        assert(self.vocabs_finalized())
        if x in self.grammar.rules_by_type:
            return self.grammar.rules_by_type[x]
        else:
            return None

    def convert(self, x:str, return_what="tensor"):     # "tensor", "ids", "actions", "tree"
        rets = [r.strip() for r in return_what.split(",")]
        ret = {}
        actions = self.grammar.actions_for(x, format=self.format)
        ret["actions"] = actions
        tree = self.grammar.actions_to_tree(actions)
        ret["tree"] = tree
        actionids = [self.vocab_actions[action] for action in actions]
        ret["ids"] = actionids
        tensor = torch.tensor(actionids, dtype=torch.long)
        ret["tensor"] = tensor
        ret = [ret[r] for r in rets]
        return ret


def try_func_query_encoder():
    fg = FuncGrammar("<R>")
    fg.add_rule("<R> -> wife :: <R>")
    fg.add_rule("<R> -> president :: <R>")
    fg.add_rule("<R> -> BO")
    fg.add_rule("<R> -> stateid :: <W>*")
    fg.add_rule("<W>* -> @W:END@")
    fg.add_rule("<W>* -> 'bo' -- <W>*")
    fg.add_rule("<W>* -> 'us' -- <W>*")
    fg.add_rule("<W>* -> 'a' -- <W>*")
    fg.add_rule("<R> -> countryid :: <W>*")
    fg.add_rule("<R> -> country :: <R>")
    fg.add_rule("<R> -> US")

    query = "wife(president(countryid('bo', 'us', 'a', @W:END@)))"
    actions = fg.actions_for(query, format="pred")
    print(actions)
    tree = fg.actions_to_tree(actions)
    print(tree)

    print(fg.rules_by_type)

    gb = FuncQueryEncoder(grammar=fg, format="pred")
    gb.finalize_vocab()

    g = gb.convert(query, return_what="tensor,tree")
    print(g[0])
    print(g[1])
    print(g[1]._action)


if __name__ == '__main__':
    try_func_query_encoder()