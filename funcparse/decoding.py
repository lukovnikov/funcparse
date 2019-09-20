from typing import Dict

import torch

from funcparse.states import FuncTreeStateBatch, StateBatch


class TransitionModel(torch.nn.Module): pass


class TFActionSeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, **kw):
        super(TFActionSeqDecoder, self).__init__(**kw)
        self.model = model

    def forward(self, fsb:FuncTreeStateBatch):
        states = fsb.unbatch()
        numex = 0
        for state in states:
            state.start_decoding()
            numex += 1

        fsb.batch()
        all_terminated = False

        step_losses = []
        step_accs = []  # 1s for correct ones, 0s for incorrect ones, -1 for masked ones
        step_terms = []

        # main loop
        while not all_terminated:
            all_terminated = True
            fsb, losses = self.model(fsb)
            step_loss, step_acc, step_term = losses
            step_losses.append(step_loss)
            step_accs.append(step_acc)
            step_terms.append(step_term)
            all_terminated = bool(torch.all(step_term == 0).item())

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





