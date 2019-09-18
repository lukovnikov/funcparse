import re
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Union, List, Dict
import qelos as q

import torch
from nltk import Tree

from funcparse.grammar import ActionTree
from funcparse.vocab import SentenceEncoder, FuncQueryEncoder


class AlignedActionTree(ActionTree):
    def __init__(self, node, children=None):
        super(AlignedActionTree, self).__init__(node, children=children)
        self._align = None

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.

        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            ret = cls(tree._label, children)
            ret._action = tree._action
            ret._align = tree._align
            return ret
        else:
            return tree


class FuncTreeState(object):
    """
    State object containing
    """
    def __init__(self, inp:str, out:str,
                 sentence_encoder:SentenceEncoder, query_encoder:FuncQueryEncoder, **kw):
        super(FuncTreeState, self).__init__(**kw)
        self.inp_string, self.out_string = inp, out
        self.sentence_encoder, self.query_encoder = sentence_encoder, query_encoder
        self.inp_tensor, self.inp_tokens = sentence_encoder.convert(inp, return_what="tensor,tokens")
        self.has_gold = False
        if out is not None:
            self.has_gold = True
            self.gold_tensor, self.gold_tree = query_encoder.convert(out, return_what="tensor,tree")
            assert(self.gold_tree.action() is not None)
        self.out_actions, self.out_tree = None, None
        self.nn_states = {"inp_tensor": self.inp_tensor} # put NN states here
        self.open_nodes = []

    @property
    def is_terminated(self):
        return len(self.open_nodes) == 0

    def get_open_nodes(self, tree=None):
        tree = tree if tree is not None else self.out_tree
        ret = []
        for child in tree:
            ret = ret + self.get_open_nodes(child)
        if tree.label() in self.query_encoder.grammar.rules_by_type:  # non terminal
            ret = ret + [tree]
        return ret

    def start_decoding(self):
        start_type = self.query_encoder.grammar.start_type
        self.out_tree = AlignedActionTree(start_type, children=[])
        self.open_nodes = [self.out_tree] + self.open_nodes
        # align to gold
        if self.has_gold:
            self.out_tree._align = self.gold_tree

    def apply_rule(self, node:AlignedActionTree, rule:Union[str, int]):
        assert(node == self.open_nodes[0])
        if isinstance(rule, str):
            ruleid = self.query_encoder.vocab_actions[rule]
            rulestr = rule
        elif isinstance(rule, int):
            ruleid = rule
            rulestr = self.query_encoder.vocab_actions(rule)

        head, body = rulestr.split(" -> ")
        func_splits = body.split(" :: ")
        sibl_splits = body.split(" -- ")

        if len(func_splits) > 1:
            rule_arg, rule_inptypes = func_splits
            rule_inptypes = rule_inptypes.split(" ")

            # replace label of tree
            node.set_label(rule_arg)
            self.open_nodes.pop(0)

            # align to gold
            if self.has_gold:
                gold_children = node._align[:]

            # create children nodes as open non-terminals
            for i, child in enumerate(rule_inptypes):
                child_node = AlignedActionTree(child, [])
                node.append(child_node)

                if self.has_gold:
                    child_node._align = gold_children[i]

            # manage open nodes
            self.open_nodes = [child_node for child_node in node if child_node.label() in self.query_encoder.grammar.rules_by_type]\
                              + self.open_nodes

        elif len(sibl_splits) > 1:
            rule_arg, rule_sibltypes = sibl_splits
            rule_sibltypes = rule_sibltypes.split(" ")
            assert(len(rule_sibltypes) == 1)

            # replace label of tree
            node.set_label(rule_arg)
            self.open_nodes.pop(0)

            # create new sibling node
            parent = node.parent()
            i = len(parent)
            new_sibl_node = AlignedActionTree(rule_sibltypes[0], [])
            parent.append(new_sibl_node)
            self.open_nodes = [new_sibl_node] if new_sibl_node.label() in self.query_encoder.grammar.rules_by_type else [] \
                                + self.open_nodes

            if self.has_gold:
                gold_child = parent._align[i]
                new_sibl_node._align = gold_child

        else:   # terminal
            node.set_label(body)
            self.open_nodes.pop(0)

    def copy_token(self, node:AlignedActionTree, inp_position:int):
        inplabel = self.inp_tokens[inp_position]
        rule = f"<W>* -> '{inplabel}' -- <W>*"
        self.apply_rule(node, rule)

    def get_valid_actions_at(self, node:AlignedActionTree):
        node_label = node.label()
        if node_label in self.query_encoder.grammar.rules_by_type:
            return self.query_encoder.grammar.rules_by_type[node_label]
        else:
            return [self.query_encoder.none_action]

    def get_gold_action_at(self, node:AlignedActionTree):
        assert(self.has_gold)
        return node._align.action()

    def apply_action(self, node:AlignedActionTree, action:str):
        copyre = re.compile("COPY\[(\d+)\]")
        if copyre.match(action):
            self.copy_token(node, int(copyre.match(action).group(1)))
        else:
            self.apply_rule(node, action)

    def to(self, device):
        self.nn_states = q.recmap(self.nn_states, lambda x: x.to(device))
        return self

    @classmethod
    def batch(self, states:List, nn_batcher=None):
        return FuncTreeStateBatch(states, nn_batcher=nn_batcher)


class StateBatch(ABC):
    @abstractmethod
    def batch(self): pass
    @abstractmethod
    def unbatch(self): pass


class NNStateBatcher(ABC):
    @abstractmethod
    def batch(self, nnstates): pass
    @abstractmethod
    def unbatch(self, batched_nnstates): pass


class DefaultNNStateBatcher(NNStateBatcher):
    def batch(self, nnstates:List[Dict[str,Union[torch.Tensor, Dict]]]):
        retkeys = set(nnstates[0].keys())
        ret = dict((k, []) for k in retkeys)
        for nnstate in nnstates:
            assert(set(nnstate.keys()) == retkeys)
            for k in retkeys:
                ret[k].append(nnstate[k])
        for k in retkeys:
            if isinstance(ret[k][0], dict): # go deeper
                ret[k] = self.batch(ret[k])
            else:
                try:
                    ret[k] = torch.stack(ret[k], 0)
                except RuntimeError as e:
                    # dimension mismatch
                    maxlen = max([retk_e.size(0) for retk_e in ret[k]])
                    for i in range(len(ret[k])):
                        r = ret[k][i]
                        repeats = [1] * r.dim()
                        repeats[0] = maxlen - r.size(0)
                        if repeats[0] != 0:
                            ret[k][i] = torch.cat([r, torch.zeros_like(r[0:1]).repeat(*repeats)], 0)
                    ret[k] = torch.stack(ret[k], 0)
        return ret

    def unbatch(self, batched_nnstates:Dict[str, Union[torch.Tensor, Dict]]):
        rets = None
        for k, v in batched_nnstates.items():
            splits = self.unbatch(v) if isinstance(v, dict) else v.split(1)
            if rets is None:
                rets = []
                for _ in splits:
                    rets.append({})
            for i, split in enumerate(splits):
                rets[i][k] = split if isinstance(v, dict) else split.squeeze(0)
        return rets


def try_default_nn_sate_batcher():
    batcher = DefaultNNStateBatcher()
    batched_state = {"x": torch.rand(2, 3, 4),
                     "y": {"a": torch.rand(2, 5),
                           "b": torch.rand(2, 4)}}
    print(batched_state)
    unbatched_states = batcher.unbatch(batched_state)
    print(unbatched_states)
    rebatched_state = batcher.batch(unbatched_states)
    print(rebatched_state)


class FuncTreeStateBatch(StateBatch):
    def __init__(self, states:List[FuncTreeState], nn_batcher:NNStateBatcher=DefaultNNStateBatcher(), **kw):
        super(FuncTreeStateBatch, self).__init__(**kw)
        self.states = [deepcopy(state) for state in states]
        self.batched_states = None
        self.nn_batcher = nn_batcher
        self.batch()

    def batch(self):    # update batched_states from states
        nnstates = [state.nn_states for state in self.states]
        self.batched_states = self.nn_batcher.batch(nnstates)

    def unbatch(self):  # update state.g's from batched_graph
        nn_states = self.nn_batcher.unbatch(self.batched_states)
        for state, nn_state in zip(self.states, nn_states):
            state.nn_state = nn_state
        return self.states

    def to(self, device):
        for state in self.states:
            state.to(device)
        self.batched_states = q.recmap(self.batched_states, lambda x: x.to(device))
        return self

    def __len__(self):
        return len(self.states)


if __name__ == '__main__':
    try_default_nn_sate_batcher()