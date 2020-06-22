import sys
_module = sys.modules[__name__]
del sys
macarico = _module
actors = _module
bow = _module
rnn = _module
annealing = _module
base = _module
data = _module
nlp_data = _module
synthetic = _module
types = _module
vocabulary = _module
features = _module
sequence = _module
lts = _module
aggrevate = _module
behavioral_cloning = _module
dagger = _module
lols = _module
ppo = _module
reinforce = _module
reslope = _module
policies = _module
bootstrap = _module
costeval = _module
linear = _module
multitask = _module
wap = _module
tasks = _module
blackjack = _module
cartpole = _module
concentration = _module
dependency_parser = _module
gridworld = _module
hexgame = _module
mdp = _module
mountain_car = _module
pendulum = _module
pocman = _module
seq2json = _module
seq2seq = _module
sequence_labeler = _module
util = _module
setup = _module
test_cartpole = _module
test_concentration = _module
test_dependency_parser = _module
test_mountain_car = _module
test_pocman = _module
test_ppo = _module
test_randomly = _module
test_seq2seq = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.autograd import Variable as Var


import numpy as np


import random


from collections import Counter


import scipy.optimize


import torch.nn


import math


import time


import itertools


from copy import deepcopy


import random as py_random


class TypeMemory(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.param = Parameter(torch.zeros(1))


def check_intentional_override(class_name, fn_name, override_bool_name, obj,
    *fn_args):
    if not getattr(obj, override_bool_name):
        try:
            getattr(obj, fn_name)(*fn_args)
        except NotImplementedError:
            print(
                """*** warning: %s %s
*** does not seem to have an implemented '%s'
*** perhaps you overrode %s by accident?
*** if not, suppress this warning by setting
*** %s=True
"""
                 % (class_name, type(obj), fn_name, fn_name[1:],
                override_bool_name), file=sys.stderr)
        except:
            pass


class Policy(nn.Module):
    """A `Policy` is any function that contains a `forward` function that
    maps states to actions."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, state):
        raise NotImplementedError('abstract')
    """
    cases where we need to reset:
    - 0. new minibatch. this means reset EVERYTHING.
    - 1. new example in a minibatch. this means reset dynamic and static _features, but not _batched_features
    - 2. replaying the current example. this means reset dynamic ONLY.
    flipped around:
    - Actors are reset in all cases
    - _features is reset in 0 and 1
    - _batched_features is reset in 0
    """

    def new_minibatch(self):
        self._reset_some(0, True)

    def new_example(self):
        self._reset_some(1, True)

    def new_run(self):
        self._reset_some(2, True)

    def _reset_some(self, reset_type, recurse):
        for module in self.modules():
            if isinstance(module, Actor):
                module.reset()
            if isinstance(module, DynamicFeatures):
                if reset_type == 0 or reset_type == 1:
                    module._features = None
                if reset_type == 0:
                    module._batched_features = None


class LearningAlg(nn.Module):

    def __call__(self, env):
        raise NotImplementedError('abstract method not defined.')


class Attention(nn.Module):
    """ It is usually the case that the `Features` one wants to compute
    are a function of only some part of the input at any given time
    step. For instance, in a sequence labeling task, one might only
    want to look at the `Features` of the word currently being
    labeled. Or in a machine translation task, one might want to have
    dynamic, differentiable softmax-style attention.

    For static `Attention`, the class must define its `arity`: the
    number of places that it looks (e.g., one in sequence labeling).
    """
    OVERRIDE_FORWARD = False
    arity = 0
    actor_dependent = False

    def __init__(self, features):
        nn.Module.__init__(self)
        self.features = features
        self.dim = (self.arity or 1) * self.features.dim
        check_intentional_override('Attention', '_forward',
            'OVERRIDE_FORWARD', self, None)

    def forward(self, state):
        fts = self._forward(state)
        dim_sum = 0
        if self.arity is None:
            assert len(fts) == 1
        if self.arity is not None:
            assert len(fts) == self.arity
        for ft in fts:
            assert ft.dim() == 2
            assert ft.shape[0] == 1
            dim_sum += ft.shape[1]
        assert dim_sum == self.dim
        return fts

    def _forward(self, state):
        raise NotImplementedError('abstract')

    def set_actor(self, actor):
        raise NotImplementedError('abstract')

    def make_out_of_bounds(self):
        oob = Parameter(torch.Tensor(1, self.features.dim))
        oob.data.zero_()
        return oob


class Torch(nn.Module):

    def __init__(self, features, dim, layers):
        nn.Module.__init__(self)
        self.features = features
        self.dim = dim
        self.torch_layers = layers if isinstance(layers, nn.ModuleList
            ) else nn.ModuleList(layers) if isinstance(layers, list
            ) else nn.ModuleList([layers])

    def forward(self, x):
        x = self.features(x)
        for l in self.torch_layers:
            x = l(x)
        return x


def Varng(*args, **kwargs):
    return torch.autograd.Variable(*args, **kwargs, requires_grad=False)


class LinearValueFn(nn.Module):

    def __init__(self, features, disconnect_values=True):
        nn.Module.__init__(self)
        self.features = features
        self.dim = features.dim
        self.disconnect_values = disconnect_values
        self.value_fn = nn.Linear(self.dim, 1)

    def forward(self, state):
        x = self.features(state)
        if self.disconnect_values:
            x = Varng(x.data)
        return self.value_fn(x)


def actions_to_probs(actions, n_actions):
    probs = torch.zeros(n_actions)
    bag_size = len(actions)
    prob = 1.0 / bag_size
    for action_set in actions:
        for action in action_set:
            probs[action] += prob / len(action_set)
    return probs


def min_set(costs, limit_actions=None):
    min_val = None
    min_set = []
    if limit_actions is None:
        for a, c in enumerate(costs):
            if min_val is None or c < min_val:
                min_val = c
                min_set = [a]
            elif c == min_val:
                min_set.append(a)
    else:
        for a in limit_actions:
            c = costs[a]
            if min_val is None or c < min_val:
                min_val = c
                min_set = [a]
            elif c == min_val:
                min_set.append(a)
    return min_set


class BootstrapCost:

    def __init__(self, costs, greedy_predict=True):
        self.costs = costs
        self.greedy_predict = greedy_predict

    def average_cost(self):
        return sum(self.costs) / len(self.costs)

    def data(self):
        if self.greedy_predict:
            return self.costs[0].data
        else:
            return self.average_cost().data

    def get_probs(self, limit_actions=None):
        assert len(self.costs) > 0
        n_actions = len(self.costs[0].data)
        actions = [min_set(c.data, limit_actions) for c in self.costs]
        return actions_to_probs(actions, n_actions)

    def __getitem__(self, idx):
        if self.greedy_predict:
            return self.costs[0][idx]
        else:
            return self.average_cost()[idx]

    def __neg__(self):
        if self.greedy_predict:
            return self.costs[0].__neg__()
        else:
            return self.average_cost().__neg__()

    def argmin(self):
        if self.greedy_predict:
            return self.costs[0].argmin()
        else:
            return self.average_cost().argmin()


def bootstrap_probabilities(n_actions, policy_bag, state, deviate_to):
    actions = [[policy(state, deviate_to)] for policy in policy_bag]
    probs = actions_to_probs(actions, n_actions)
    return probs


def build_policy_bag(features_bag, n_actions, loss_fn, n_layers, hidden_dim):
    return [LinearPolicy(features, n_actions, loss_fn=loss_fn, n_layers=
        n_layers, hidden_dim=hidden_dim) for features in features_bag]


def delegate_with_poisson(params, functions, greedy_update):
    total_loss = 0.0
    functions_params_pairs = zip(functions, params)
    for idx, (loss_fn, params) in enumerate(functions_params_pairs):
        loss_i = loss_fn(*params)
        if greedy_update and idx == 0:
            count_i = 1
        else:
            count_i = np.random.poisson(1)
        total_loss = total_loss + count_i * loss_i
    return total_loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hal3_macarico(_paritybench_base):
    pass
    def test_000(self):
        self._check(Torch(*[], **{'features': ReLU(), 'dim': 4, 'layers': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

