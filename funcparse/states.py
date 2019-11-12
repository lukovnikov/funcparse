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
    orderless = ["and", "or"]
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

    def eq(self, other):
        assert(isinstance(other, ActionTree))
        if self._label != other._label:
            return False
        if self._label in self.orderless:
            # check if every child is in other and other contains no more
            if len(self) != len(other):
                return False
            selfchildren = [selfchild for selfchild in self]
            otherchildren = [otherchild for otherchild in other]
            if not selfchildren[-1].eq(otherchildren[-1]):
                return False    # terminator must be same and in the end
            else:
                selfchildren = selfchildren[:-1]
                otherchildren = otherchildren[:-1]
            i = 0
            while i < len(selfchildren):
                selfchild = selfchildren[i]
                j = 0
                unbroken = True
                while j < len(otherchildren):
                    otherchild = otherchildren[j]
                    if selfchild.eq(otherchild):
                        selfchildren.pop(i)
                        otherchildren.pop(j)
                        i -= 1
                        j -= 1
                        unbroken = False
                        break
                    j += 1
                if unbroken:
                    return False
                i += 1
            if len(selfchildren) == 0 and len(otherchildren) == 0:
                return True
            else:
                return False
        else:
            return all([selfchild.eq(otherchild) for selfchild, otherchild in zip(self, other)])





class FuncTreeState(object):
    """
    State object containing
    """
    def __init__(self, inp:str, out:str=None,
                 sentence_encoder:SentenceEncoder=None, query_encoder:FuncQueryEncoder=None, **kw):
        super(FuncTreeState, self).__init__(**kw)
        self.inp_string, self.out_string = inp, out
        self.sentence_encoder, self.query_encoder = sentence_encoder, query_encoder
        self.inp_tensor = None
        if sentence_encoder is not None:
            self.inp_tensor, self.inp_tokens = sentence_encoder.convert(inp, return_what="tensor,tokens")

        self.has_gold = False
        self.use_gold = False
        if out is not None:
            self.has_gold = True
            self.use_gold = self.has_gold
            if query_encoder is not None:
                self.gold_tensor, self.gold_tree, self.gold_rules = query_encoder.convert(out, return_what="tensor,tree,actions")
                assert(self.gold_tree.action() is not None)
        self.out_tree = None
        self.out_rules = None
        if self.inp_tensor is not None:
            self.nn_states = {"inp_tensor": self.inp_tensor} # put NN states here
        self.open_nodes = []

    def make_copy(self):
        ret = type(self)(self.inp_string, self.out_string)
        for k, v in self.__dict__.items():
            ret.__dict__[k] = deepcopy(v) if k not in ["sentence_encoder", "query_encoder"] else v
        ret.open_nodes = ret.get_open_nodes()
        return ret

    def reset(self):
        self.open_nodes = []
        self.nn_states = {"inp_tensor": self.inp_tensor}
        self.out_tree = None
        self.out_rules = None
        return self

    @property
    def is_terminated(self):
        return len(self.open_nodes) == 0

    def get_open_nodes(self, tree=None):
        if self.out_tree is None and tree is None:
            return []
        tree = tree if tree is not None else self.out_tree
        ret = []
        for child in tree:
            ret = ret + self.get_open_nodes(child)
        if tree.label() in self.query_encoder.grammar.rules_by_type:  # non terminal
            ret = ret + [tree]
        return ret

    def start_decoding(self):
        self.reset()
        start_type = self.query_encoder.grammar.start_type
        self.out_tree = AlignedActionTree(start_type, children=[])
        self.out_rules = []
        self.open_nodes = [self.out_tree] + self.open_nodes
        # align to gold
        if self.use_gold:
            self.out_tree._align = self.gold_tree
        # initialize prev_action
        self.nn_states["prev_action"] = torch.tensor(self.query_encoder.vocab_actions[self.query_encoder.start_action],
                                                     device=self.nn_states["inp_tensor"].device, dtype=torch.long)

    def apply_rule(self, node:AlignedActionTree, rule:Union[str, int]):
        # if node.label() not in self.query_encoder.grammar.rules_by_type \
        #         or rule not in self.query_encoder.grammar.rules_by_type[node.label()]:
        #     raise Exception("something wrong")
        #     return
        self.nn_states["prev_action"] = torch.ones_like(self.nn_states["prev_action"]) \
                                        * self.query_encoder.vocab_actions[rule]
        self.out_rules.append(rule)
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

        if len(sibl_splits) > 1:
            raise Exception("sibling rules no longer supported")

        self.open_nodes.pop(0)

        if node.label()[-1] in "*+" and body != f"{head}:END@":  # variable number of children
            # create new sibling node
            parent = node.parent()
            i = len(parent)

            new_sibl_node = AlignedActionTree(node.label(), [])
            parent.append(new_sibl_node)

            # manage open nodes
            self.open_nodes = ([new_sibl_node]
                               if (new_sibl_node.label() in self.query_encoder.grammar.rules_by_type
                                   or new_sibl_node.label()[:-1] in self.query_encoder.grammar.rules_by_type)
                               else []) \
                              + self.open_nodes

            if self.use_gold:
                gold_child = parent._align[i]
                new_sibl_node._align = gold_child

        if len(func_splits) > 1 :
            rule_arg, rule_inptypes = func_splits
            rule_inptypes = rule_inptypes.split(" ")

            # replace label of tree
            node.set_label(rule_arg)
            node.set_action(rule)

            # align to gold
            if self.use_gold:
                gold_children = node._align[:]

            # create children nodes as open non-terminals
            for i, child in enumerate(rule_inptypes):
                child_node = AlignedActionTree(child, [])
                node.append(child_node)

                if self.use_gold:
                    child_node._align = gold_children[i]

            # manage open nodes
            self.open_nodes = [child_node for child_node in node if child_node.label() in self.query_encoder.grammar.rules_by_type]\
                              + self.open_nodes
        else:   # terminal
            node.set_label(body)
            node.set_action(rule)

    def copy_token(self, node:AlignedActionTree, inp_position:int):
        inplabel = self.inp_tokens[inp_position]
        rule = f"<W> -> '{inplabel}'"
        self.apply_rule(node, rule)

    def get_valid_actions_at(self, node:AlignedActionTree):
        action_mask = self.get_valid_action_mask_at(node)
        valid_action_ids = action_mask.nonzero().cpu().numpy()
        # TODO: finish: translate back to strings

        # node_label = node.label()
        # if node_label in self.query_encoder.grammar.rules_by_type:
        #     if node_label[-1] in "*+":
        #         ret = self.query_encoder.grammar.rules_by_type[node_label]
        #         ret += self.query_encoder.grammar.rules_by_type[node_label[:-1]]
        #         return ret
        #     else:
        #         return self.query_encoder.grammar.rules_by_type[node_label]
        # else:
        #     return [self.query_encoder.none_action]

    def get_valid_action_mask_at(self, node:AlignedActionTree):
        node_label = node.label()
        ret = self.query_encoder.get_action_mask_for(node_label)
        return ret

    def get_gold_action_at(self, node:AlignedActionTree):
        assert(self.use_gold)
        return node._align.action()

    def apply_action(self, node:AlignedActionTree, action:str):
        # self.out_actions.append(action)
        copyre = re.compile("COPY\[(\d+)\]")
        if copyre.match(action):
            self.copy_token(node, int(copyre.match(action).group(1)))
        else:
            self.apply_rule(node, action)

    def to(self, device):
        self.nn_states = q.recmap(self.nn_states, lambda x: x.to(device))
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
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
            splits = self.unbatch(v) if isinstance(v, dict) else v.split(1, 0)
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
        self.states = [state.make_copy() for state in states]
        self.batched_states = None
        self.nn_batcher = nn_batcher
        self.batch()

    def batch(self):    # update batched_states from states
        nnstates = [state.nn_states for state in self.states]
        self.batched_states = self.nn_batcher.batch(nnstates)

    def unbatch(self):  # update state.g's from batched_graph
        nn_states = self.nn_batcher.unbatch(self.batched_states)
        for state, nn_state in zip(self.states, nn_states):
            state.nn_states = nn_state
        return self.states

    def to(self, device):
        for state in self.states:
            state.to(device)
        self.batched_states = q.recmap(self.batched_states, lambda x: x.to(device))
        return self

    def __len__(self):
        return len(self.states)

    def make_copy(self):
        ret = type(self)(self.states, self.nn_batcher)
        return ret

    def new(self, states:List[FuncTreeState]):
        ret = type(self)(states, nn_batcher=self.nn_batcher)
        return ret


if __name__ == '__main__':
    try_default_nn_sate_batcher()