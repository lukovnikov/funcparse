from typing import Dict

import torch
import qelos as q
import numpy as np

from funcparse.states import FuncTreeStateBatch, StateBatch


class TransitionModel(torch.nn.Module): pass


class TFActionSeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, smoothing=0., **kw):
        super(TFActionSeqDecoder, self).__init__(**kw)
        self.model = model
        if smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, mode="probs")
        else:
            self.loss = q.CELoss(reduction="none", ignore_index=0, mode="probs")

    def forward(self, fsb:FuncTreeStateBatch):
        states = fsb.unbatch()
        numex = 0
        for state in states:
            state.start_decoding()
            numex += 1

        all_terminated = False

        step_losses = []
        step_accs = []  # 1s for correct ones, 0s for incorrect ones, -1 for masked ones
        step_terms = []

        # main loop
        while not all_terminated:
            fsb.batch()
            probs, fsb = self.model(fsb)
            states = fsb.unbatch()
            # find gold rules
            gold_rules = torch.zeros(numex, device=probs.device, dtype=torch.long)
            term_mask = torch.zeros_like(gold_rules).float()
            for i, state in enumerate(states):
                if not state.is_terminated:
                    open_node = state.open_nodes[0]
                    gold_rule_str = state.get_gold_action_at(open_node)
                    gold_rules[i] = state.query_encoder.vocab_actions[gold_rule_str]
                    term_mask[i] = 1
                    state.apply_action(open_node, gold_rule_str)
            # compute loss
            step_loss = self.loss(probs, gold_rules)
            # compute accuracy
            _, pred_rules = probs.max(-1)
            step_acc = (pred_rules == gold_rules).float()
            step_losses.append(step_loss)
            step_accs.append(step_acc)
            step_terms.append(term_mask)
            all_terminated = bool(torch.all(term_mask == 0).item())

        mask = torch.stack(step_terms, 0).transpose(1, 0)
        # aggregate losses
        losses = torch.stack(step_losses, 0).transpose(1, 0)
        losses = losses * mask.float()
        loss = losses.sum(-1).mean()

        # aggregate accuracies
        step_accs = torch.stack(step_accs, 0).transpose(1, 0)
        elemacc = step_accs.float().sum() / mask.float().sum()
        seqacc = (step_accs.bool() | ~mask.bool()).all(-1).float().mean()

        return {"output": fsb, "loss": loss, "any_acc": elemacc, "seq_acc": seqacc}


class BeamActionSeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, beamsize=1, maxsteps=25, **kw):
        super(BeamActionSeqDecoder, self).__init__(**kw)
        self.model = model
        self.beamsize = beamsize
        self.maxsteps = maxsteps

    def forward(self, fsb:FuncTreeStateBatch):
        hasgold = []
        with torch.no_grad():
            fsb_original = fsb
            fsb = fsb_original.make_copy()
            states = fsb.unbatch()
            numex = 0
            for state in states:
                hasgold.append(state.has_gold)
                state.use_gold = False  # disable gold
                state.start_decoding()
                numex += 1

            assert(all([_hg is True for _hg in hasgold]) or all([_hg is False for _hg in hasgold]))
            hasgold = hasgold[0]

            all_terminated = False

            beam_batches = None

            step = 0
            while not all_terminated:
                all_terminated = True
                if beam_batches is None:    # first time
                    fsb.batch()
                    probs, fsb = self.model(fsb)
                    fsb.unbatch()
                    best_probs, best_actions = (-torch.log(probs)).topk(self.beamsize, -1, largest=False)    # (batsize, beamsize) scores and action ids, sorted
                    if (best_probs == np.infty).any():
                        print("whut")
                    beam_batches = [fsb.make_copy() for _ in range(self.beamsize)]
                    best_actions_ = best_actions.cpu().numpy()
                    for i, beam_batch in enumerate(beam_batches):
                        for j, state in enumerate(beam_batch.states):
                            if not state.is_terminated:
                                open_node = state.open_nodes[0]
                                action_str = state.query_encoder.vocab_actions(best_actions_[j, i])
                                state.apply_action(open_node, action_str)
                                all_terminated = False
                else:
                    out_beam_batches = []
                    out_beam_actions = []
                    out_beam_probs = []
                    for k, beam_batch in enumerate(beam_batches):
                        beam_batch.batch()
                        beam_probs, beam_batch = self.model(beam_batch)
                        beam_batch.unbatch()
                        beam_best_probs, beam_best_actions = (-torch.log(beam_probs)).topk(self.beamsize, -1, largest=False)
                        out_beam_probs.append(beam_best_probs + best_probs[:, k:k+1])
                        out_beam_batches.append([k]*self.beamsize)
                        out_beam_actions.append(beam_best_actions)
                    out_beam_probs = torch.cat(out_beam_probs, 1)
                    out_beam_batches = [xe for x in out_beam_batches for xe in x]
                    out_beam_actions = torch.cat(out_beam_actions, 1)

                    beam_best_probs, beam_best_k = torch.topk(out_beam_probs, self.beamsize, -1, largest=False)
                    out_beam_actions_ = out_beam_actions.cpu().numpy()
                    beam_best_k_ = beam_best_k.cpu().numpy()
                    new_beam_batches = []
                    for i in range(beam_best_k.shape[1]):
                        _state_batch = []
                        for j in range(beam_best_k.shape[0]):
                            _state = beam_batches[out_beam_batches[beam_best_k_[j, i]]].states[j]
                            _state = _state.make_copy()
                            if not _state.is_terminated:
                                all_terminated = False
                                open_node = _state.open_nodes[0]
                                action_id = out_beam_actions_[j, beam_best_k_[j, i]]
                                action_str = _state.query_encoder.vocab_actions(action_id)
                                _state.apply_action(open_node, action_str)
                            _state_batch.append(_state)
                        _state_batch = fsb.new(_state_batch)
                        new_beam_batches.append(_state_batch)
                    beam_batches = new_beam_batches
                    best_probs = beam_best_probs
                step += 1
                if step >= self.maxsteps:
                    break
            pass

        all_out_states = beam_batches       # states for whole beam
        all_out_probs = best_probs

        best_out_states = all_out_states[0]
        best_out_probs = all_out_probs[0]

        if hasgold:     # compute accuracies (seq and tree) with top scoring states
            seqaccs = 0
            treeaccs = 0
            # elemaccs = 0
            total = 0
            for best_out_state in best_out_states.states:
                seqaccs += float(best_out_state.out_rules == best_out_state.gold_rules)
                treeaccs += float(str(best_out_state.out_tree) == str(best_out_state.gold_tree))
                # elemaccs += float(best_out_state.out_rules[:len(best_out_state.gold_rules)] == best_out_state.gold_rules)
                total += 1
            return {"output": best_out_states, "output_probs": best_out_probs,
                    "seq_acc": seqaccs/total, "tree_acc": treeaccs/total}
        else:
            return {"output": all_out_states, "probs": all_out_probs}



class LSTMCellTransition(TransitionModel):
    def __init__(self, *cells:torch.nn.LSTMCell, dropout:float=0., **kw):
        super(LSTMCellTransition, self).__init__(**kw)
        self.cells = torch.nn.ModuleList(cells)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inp:torch.Tensor, states:Dict[str,torch.Tensor]):
        x = inp
        for i in range(len(self.cells)):
            _x = self.dropout(x)
            if f"{i}" not in states:
                x, c = self.cells[i](_x, None)
                states[f"{i}"] = {}
                # intialize dropout
                h_dropout_mask = self.dropout(torch.ones(inp.size(0), self.cells[i].hidden_size, device=inp.device))
                c_dropout_mask = self.dropout(torch.ones_like(h_dropout_mask))
                states[f"{i}"]["h.dropout"] = h_dropout_mask
                states[f"{i}"]["c.dropout"] = c_dropout_mask
            else:
                x, c = self.cells[i](_x, (states[f"{i}"]["h"] * states[f"{i}"]["h.dropout"], states[f"{i}"]["c"] * states[f"{i}"]["c.dropout"]))
            states[f"{i}"]["h"] = x
            states[f"{i}"]["c"] = c
        return x





