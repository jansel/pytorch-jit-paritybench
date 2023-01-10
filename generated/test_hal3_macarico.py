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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.autograd import Variable as Var


from collections import Counter


import numpy as np


import random


import scipy.optimize


import torch.nn


import math


from collections import defaultdict


import time


from collections import deque


import itertools


from copy import deepcopy


import random as py_random


class TypeMemory(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.param = Parameter(torch.zeros(1))


def check_intentional_override(class_name, fn_name, override_bool_name, obj, *fn_args):
    if not getattr(obj, override_bool_name):
        try:
            getattr(obj, fn_name)(*fn_args)
        except NotImplementedError:
            None
        except:
            pass


class DynamicFeatures(nn.Module):
    """`DynamicFeatures` are any function that map an `Env` to a
    tensor. The dimension of the feature representation tensor should
    be (1, N, `dim`), where `N` is the length of the input, and
    `dim()` returns the dimensionality.

    The `forward` function computes the features."""
    OVERRIDE_FORWARD = False

    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self._current_env = None
        self._features = None
        self._batched_features = None
        self._batched_lengths = None
        self._my_id = '%s #%d' % (type(self), id(self))
        self._recompute_always = True
        self._typememory = TypeMemory()
        check_intentional_override('DynamicFeatures', '_forward', 'OVERRIDE_FORWARD', self, None)

    def _forward(self, env):
        raise NotImplementedError('abstract')

    def forward(self, env):
        if self._batched_features is not None and hasattr(env, '_stored_batch_features') and self._my_id in env._stored_batch_features:
            i = env._stored_batch_features[self._my_id]
            assert 0 <= i and i < self._batched_features.shape[0]
            assert self._batched_lengths[i] <= self._batched_features.shape[1]
            l = self._batched_lengths[i]
            self._features = self._batched_features[i, :l, :].unsqueeze(0)
        if self._features is None or self._recompute_always:
            self._features = self._forward(env)
            assert self._features.dim() == 3
            assert self._features.shape[0] == 1
            assert self._features.shape[2] == self.dim
        assert self._features is not None
        return self._features

    def _forward_batch(self, envs):
        raise NotImplementedError('abstract')

    def forward_batch(self, envs):
        if self._batched_features is not None:
            return self._batched_features, self._batched_lengths
        try:
            res = self._forward_batch(envs)
            assert isinstance(res, tuple)
            self._batched_features, self._batched_lengths = res
            if self._batched_features.shape[0] != len(envs):
                ipdb.set_trace()
        except NotImplementedError:
            pass
        if self._batched_features is None:
            bf = [self._forward(env) for env in envs]
            for x in bf:
                assert x.dim() == 3
                assert x.shape[0] == 1
                assert x.shape[2] == self.dim
            max_len = max(x.shape[1] for x in bf)
            self._batched_features = Var(self._typememory.param.data.new(len(envs), max_len, self.dim).zero_())
            self._batched_lengths = []
            for i, x in enumerate(bf):
                self._batched_features[i, :x.shape[1], :] = x
                self._batched_lengths.append(x.shape[1])
        assert self._batched_features.shape[0] == len(envs)
        for i, env in enumerate(envs):
            if not hasattr(env, '_stored_batch_features'):
                env._stored_batch_features = dict()
            env._stored_batch_features[self._my_id] = i
        return self._batched_features, self._batched_lengths


class StaticFeatures(DynamicFeatures):

    def __init__(self, dim):
        DynamicFeatures.__init__(self, dim)
        self._recompute_always = False


class Actor(nn.Module):
    """An `Actor` is a module that computes features dynamically as a policy runs."""
    OVERRIDE_FORWARD = False

    def __init__(self, n_actions, dim, attention):
        nn.Module.__init__(self)
        self._current_env = None
        self._features = None
        self.n_actions = n_actions
        self.dim = dim
        self.attention = nn.ModuleList(attention)
        self._T = None
        self._last_t = 0
        for att in attention:
            if att.actor_dependent:
                att.set_actor(self)
        self._typememory = TypeMemory()
        check_intentional_override('Actor', '_forward', 'OVERRIDE_FORWARD', self, None)

    def reset(self):
        self._last_t = 0
        self._T = None
        self._features = None
        self._reset()

    def _reset(self):
        pass

    def _forward(self, state, x):
        raise NotImplementedError('abstract')

    def hidden(self):
        raise NotImplementedError('abstract')

    def forward(self, env):
        if self._features is None or self._T is None:
            self._T = env.horizon()
            self._features = [None] * self._T
            self._last_t = 0
        t = env.timestep()
        if t > self._last_t + 1:
            ipdb.set_trace()
        assert t <= self._last_t + 1, '%d <= %d+1' % (t, self._last_t)
        assert t >= self._last_t, '%d >= %d' % (t, self._last_t)
        self._last_t = t
        assert self._features is not None
        assert t >= 0, 'expect t>=0, bug?'
        assert t < self._T, '%d=t < T=%d' % (t, self._T)
        assert t < len(self._features)
        if self._features[t] is not None:
            return self._features[t]
        assert t == 0 or self._features[t - 1] is not None
        x = []
        for att in self.attention:
            x += att(env)
        ft = self._forward(env, x)
        assert ft.dim() == 2
        assert ft.shape[0] == 1
        assert ft.shape[1] == self.dim
        self._features[t] = ft
        return self._features[t]


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


class StochasticPolicy(Policy):

    def stochastic(self, state):
        raise NotImplementedError('abstract')

    def sample(self, state):
        return self.stochastic(state)[0]


class Example(object):

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        self.Yhat = None

    def __str__(self):
        return '{ X: %s, Y: %s, Yhat: %s }' % (self.X, self.Y, self.Yhat)

    def __repr__(self):
        return str(self)

    def input_str(self):
        return self._simple_str(self.X)

    def output_str(self):
        return self._simple_str(self.Y)

    def prediction_str(self):
        return self._simple_str(self.Yhat)

    def _simple_str(self, A):
        if A is None:
            return '?'
        if isinstance(A, list):
            return ' '.join(map(str, A))
        return str(A)


class Env(object):
    """An implementation of an environment; aka a search task or MDP.

    Args:
        n_actions: the number of unique actions available to a policy
                   in this Env (actions are numbered [0, n_actions))

    Must provide a `_run_episode(policy)` function that performs a
    complete run through this environment, acting according to
    `policy`.

    May optionally provide a `_rewind` function that some learning
    algorithms (e.g., LOLS) requires.
    """
    OVERRIDE_RUN_EPISODE = False
    OVERRIDE_REWIND = False

    def __init__(self, n_actions, T, example=None):
        self.n_actions = n_actions
        self.T = T
        self.example = Example() if example is None else example
        self._trajectory = []

    def horizon(self):
        return self.T

    def timestep(self):
        return len(self._trajectory)

    def output(self):
        return self._trajectory

    def run_episode(self, policy):

        def _policy(state):
            assert self.timestep() < self.horizon()
            a = policy(state)
            self._trajectory.append(a)
            return a
        if hasattr(policy, 'new_example'):
            policy.new_example()
        self.rewind(policy)
        _policy.new_minibatch = policy.new_minibatch if hasattr(policy, 'new_minibatch') else None
        _policy.new_example = policy.new_example if hasattr(policy, 'new_example') else None
        _policy.new_run = policy.new_run if hasattr(policy, 'new_run') else None
        out = self._run_episode(_policy)
        self.example.Yhat = out if out is not None else self._trajectory
        return self.example.Yhat

    def input_x(self):
        return self.example.X

    def rewind(self, policy):
        self._trajectory = []
        if hasattr(policy, 'new_run'):
            policy.new_run()
        self._rewind()

    def _run_episode(self, policy):
        raise NotImplementedError('abstract')

    def _rewind(self):
        raise NotImplementedError('abstract')


class CostSensitivePolicy(Policy):
    OVERRIDE_UPDATE = False

    def __init__(self):
        nn.Module.__init__(self)
        check_intentional_override('CostSensitivePolicy', '_update', 'OVERRIDE_UPDATE', self, None, None)

    def predict_costs(self, state):
        raise NotImplementedError('abstract')

    def costs_to_action(self, state, pred_costs):
        if isinstance(pred_costs, Var):
            pred_costs = pred_costs.data
        if state.actions is None or len(state.actions) == 0 or len(state.actions) == pred_costs.shape[0]:
            return pred_costs.min(0)[1][0]
        i = None
        for a in state.actions:
            if i is None or pred_costs[a] < pred_costs[i]:
                i = a
        return i

    def update(self, state_or_pred_costs, truth, actions=None):
        if isinstance(state_or_pred_costs, Env):
            assert actions is None
            actions = state_or_pred_costs.actions
            state_or_pred_costs = self.predict_costs(state_or_pred_costs)
        return self._update(state_or_pred_costs, truth, actions)

    def _update(self, pred_costs, truth, actions=None):
        raise NotImplementedError('abstract')


class Learner(Policy):
    """A `Learner` behaves identically to a `Policy`, but does "stuff"
    internally to, eg., compute gradients through pytorch's `backward`
    procedure. Not all learning algorithms can be implemented this way
    (e.g., LOLS) but most can (DAgger, reinforce, etc.)."""

    def forward(self, state):
        raise NotImplementedError('abstract method not defined.')

    def get_objective(self, loss):
        raise NotImplementedError('abstract method not defined.')


class NoopLearner(Learner):

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state):
        return self.policy(state)

    def get_objective(self, loss):
        return 0.0


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
        check_intentional_override('Attention', '_forward', 'OVERRIDE_FORWARD', self, None)

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
        self.torch_layers = layers if isinstance(layers, nn.ModuleList) else nn.ModuleList(layers) if isinstance(layers, list) else nn.ModuleList([layers])

    def forward(self, x):
        x = self.features(x)
        for l in self.torch_layers:
            x = l(x)
        return x


def Varng(*args, **kwargs):
    return torch.autograd.Variable(*args, **kwargs, requires_grad=False)


class EmbeddingFeatures(macarico.StaticFeatures):

    def __init__(self, n_types, input_field='X', d_emb=50, initial_embeddings=None, learn_embeddings=True):
        d_emb = d_emb or initial_embeddings.shape[1]
        macarico.StaticFeatures.__init__(self, d_emb)
        self.input_field = input_field
        self.initial_embeddings = None
        self.learned_embeddings = None
        if initial_embeddings is None:
            assert learn_embeddings, 'must either be learning embeddings or have initial_embeddings != None'
            self.learned_embeddings = nn.Embedding(n_types, d_emb)
        else:
            e0_v, e0_d = initial_embeddings.shape
            assert e0_v == n_types, 'got initial_embeddings with first dim=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, 'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            self.initial_embeddings = nn.Embedding(n_types, d_emb)
            self.initial_embeddings.weight.data.copy_(torch.from_numpy(initial_embeddings))
            self.initial_embeddings.requires_grad = False
            if learn_embeddings:
                self.learned_embeddings = nn.Embedding(n_types, d_emb)
                self.learned_embeddings.weight.data.zero_()

    def embed(self, txt_var):
        if self.initial_embeddings is None:
            return self.learned_embeddings(txt_var)
        if self.learned_embeddings is None:
            return self.initial_embeddings(txt_var)
        return self.learned_embeddings(txt_var) + self.initial_embeddings(txt_var)

    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        return self.embed(Varng(util.longtensor(self, txt))).view(1, -1, self.dim)

    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        x = util.longtensor(self, batch_size, max_len).zero_()
        for n, txt in enumerate(txts):
            for i in range(txt_len[n]):
                x[n, i] = int(txt[i])
        return self.embed(Varng(x)).view(batch_size, max_len, self.dim), txt_len


class BOWFeatures(macarico.StaticFeatures):

    def __init__(self, n_types, input_field='X', window_size=0, hashing=False):
        dim = (1 + 2 * window_size) * n_types
        macarico.StaticFeatures.__init__(self, dim)
        self.n_types = n_types
        self.input_field = input_field
        self.window_size = window_size
        self.hashing = hashing

    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        bow = util.zeros(self, 1, len(txt), self.dim)
        self.set_bow(bow, 0, txt)
        return Varng(bow)

    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        bow = util.zeros(self, batch_size, max_len, self.dim)
        for j, txt in enumerate(txts):
            self.set_bow(bow, j, txt)
        return Varng(bow), txt_len

    def set_bow(self, bow, j, txt):
        for n, word in enumerate(txt):
            for i in range(-self.window_size, self.window_size + 1):
                m = n + i
                if m < 0:
                    continue
                if m >= len(txt):
                    continue
                v = (i + self.window_size) * self.n_types + self.hashit(word)
                bow[j, m, v] = 1

    def hashit(self, word):
        if self.hashing:
            word = (word + 48193471) * 849103817
        return int(word % self.n_types)


qrnn_available = False


class RNN(macarico.StaticFeatures):

    def __init__(self, features, d_rnn=50, bidirectional=True, n_layers=1, cell_type='LSTM', dropout=0.0, qrnn_use_cuda=False, *extra_rnn_args):
        macarico.StaticFeatures.__init__(self, d_rnn * (2 if bidirectional else 1))
        self.features = features
        self.bidirectional = bidirectional
        self.d_emb = features.dim
        self.d_rnn = d_rnn
        assert cell_type in ['LSTM', 'GRU', 'RNN', 'QRNN']
        if cell_type == 'QRNN':
            assert qrnn_available, 'you asked from QRNN but torchqrnn is not installed'
            assert dropout == 0.0, 'QRNN does not support dropout'
            self.rnn = QRNN(self.d_emb, self.d_rnn, *extra_rnn_args, num_layers=n_layers, use_cuda=qrnn_use_cuda)
            if bidirectional:
                self.rnn2 = QRNN(self.d_emb, self.d_rnn, *extra_rnn_args, num_layers=n_layers, use_cuda=qrnn_use_cuda)
                self.rev = list(range(255, -1, -1))
        else:
            self.rnn = getattr(nn, cell_type)(self.d_emb, self.d_rnn, *extra_rnn_args, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)

    def _forward(self, env):
        e = self.features(env)
        [res, _] = self.rnn(e)
        if hasattr(self, 'rnn2'):
            [res2, _] = self.rnn2(e[:, self.rev[-e.shape[1]:], :])
            res = torch.cat([res, res2[:, self.rev[-res2.shape[1]:], :]], dim=2)
        return res

    def inv_perm(self, perm):
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse

    def _forward_batch(self, envs):
        e, lens = self.features.forward_batch(envs)
        sort_idx = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        e = e[sort_idx]
        sort_lens = [lens[i] for i in sort_idx]
        pack = torch.nn.utils.rnn.pack_padded_sequence(e, sort_lens, batch_first=True)
        [r, _] = self.rnn(e)
        r = r[self.inv_perm(sort_idx)]
        if hasattr(self, 'rnn2'):
            [r2, _] = self.rnn2(e[:, self.rev[-e.shape[1]:], :])
            r = torch.cat([r, r2[:, self.rev[-r2.shape[1]:], :]], dim=2)
        return r, lens


class DilatedCNN(macarico.StaticFeatures):
    """see https://arxiv.org/abs/1702.02098"""

    def __init__(self, features, n_layers=4, passthrough=True):
        macarico.StaticFeatures.__init__(self, features.dim)
        self.features = features
        self.passthrough = passthrough
        self.conv = nn.ModuleList([nn.Linear(3 * self.dim, self.dim) for i in range(n_layers)])
        self.oob = Parameter(torch.Tensor(self.dim))

    def _forward(self, env):
        l1, l2 = (0.5, 0.5) if self.passthrough else (0, 1)
        X = self.features(env).squeeze(0)
        N = X.shape[0]
        get = lambda XX, n: self.oob if n < 0 or n >= N else XX[n]
        dilation = [(2 ** n) for n in range(len(self.conv) - 1)] + [1]
        for delta, lin in zip(dilation, self.conv):
            X = [(l1 * get(X, n) + l2 * F.relu(lin(torch.cat([get(X, n), get(X, n - delta), get(X, n + delta)], dim=0)))) for n in range(N)]
        return torch.cat(X, 0).view(1, N, self.dim)


class AverageAttention(macarico.Attention):
    arity = None

    def __init__(self, features):
        macarico.Attention.__init__(self, features)

    def _forward(self, state):
        x = self.features(state)
        return [x.mean(dim=1)]


class AttendAt(macarico.Attention):
    """Attend to the current token's *input* embedding.
    """
    arity = 1

    def __init__(self, features, position='n'):
        macarico.Attention.__init__(self, features)
        self.position = (lambda state: getattr(state, position)) if isinstance(position, str) else position if hasattr(position, '__call__') else None
        assert self.position is not None
        self.oob = self.make_out_of_bounds()

    def _forward(self, state):
        x = self.features(state)
        n = self.position(state)
        if n < 0:
            return [self.oob]
        if n >= x.shape[1]:
            return [self.oob]
        return [x[0, n].unsqueeze(0)]


class FrontBackAttention(macarico.Attention):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2

    def __init__(self, features):
        macarico.Attention.__init__(self, features)

    def _forward(self, state):
        x = self.features(state)
        return [x[0, 0].unsqueeze(0), x[0, -1].unsqueeze(0)]


class SoftmaxAttention(macarico.Attention):
    arity = None
    actor_dependent = True

    def __init__(self, features, bilinear=True):
        macarico.Attention.__init__(self, features)
        self.bilinear = bilinear
        self.actor = [None]

    def set_actor(self, actor):
        assert self.actor[0] is None
        self.actor[0] = actor
        self.attention = nn.Bilinear(self.actor[0].dim, self.features.dim, 1) if self.bilinear else nn.Linear(self.actor[0].dim + self.features.dim, 1)

    def _forward(self, state):
        x = self.features(state).squeeze(0)
        h = self.actor[0].hidden()
        N = x.shape[0]
        alpha = self.attention(h.repeat(N, 1), x) if self.bilinear else self.attention(torch.cat([h.repeat(N, 1), x], 1))
        return [F.softmax(alpha.view(1, N), dim=1).mm(x)]


class Annealing:
    """Base case."""

    def __call__(self, T):
        raise NotImplementedError('abstract method not implemented.')


class NoAnnealing(Annealing):
    """Constant rate."""

    def __init__(self, value):
        self.value = value

    def __call__(self, T):
        return self.value


def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = torch.zeros(state.n_actions)
    set_costs = False
    try:
        reference.set_min_costs_to_go(state, costs)
        set_costs = True
    except NotImplementedError:
        pass
    if not set_costs:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    old_actions = state.actions
    min_cost = min(costs[a] for a in old_actions)
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a


class stochastic(object):

    def __init__(self, inst):
        assert isinstance(inst, Annealing)
        self.inst = inst
        self.time = 1

    def step(self):
        self.time += 1

    def __call__(self):
        return random() <= self.inst(self.time)


class AggreVaTe(macarico.Learner):

    def __init__(self, policy, reference, p_rollin_ref=NoAnnealing(0)):
        macarico.Learner.__init__(self)
        assert isinstance(policy, CostSensitivePolicy)
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        pred_costs = self.policy.predict_costs(state)
        costs = torch.zeros(self.policy.n_actions)
        try:
            costs.zero_()
            self.reference.set_min_costs_to_go(state, costs)
        except NotImplementedError:
            raise ValueError('can only run aggrevate on reference losses that define min_cost_to_go; try lols with rollout=ref instead')
        costs -= costs.min()
        self.objective += self.policy.update(pred_costs, costs, state.actions)
        return break_ties_by_policy(self.reference, self.policy, state, False) if self.rollin_ref() else self.policy.costs_to_action(state, pred_costs)

    def get_objective(self, _):
        ret = self.objective
        self.objective = 0.0
        self.rollin_ref.step()
        return ret


class DAgger(macarico.Learner):

    def __init__(self, policy, reference, p_rollin_ref=NoAnnealing(0)):
        macarico.Learner.__init__(self)
        self.rollin_ref = stochastic(p_rollin_ref)
        self.policy = policy
        self.reference = reference
        self.objective = 0.0

    def forward(self, state):
        ref = break_ties_by_policy(self.reference, self.policy, state, False)
        pol = self.policy(state)
        self.objective += self.policy.update(state, ref)
        return ref if self.rollin_ref() else pol

    def get_objective(self, _):
        ret = self.objective
        self.objective = 0.0
        self.rollin_ref.step()
        return ret


def argmin(vec, allowed=None, dim=0):
    if isinstance(vec, Var):
        vec = vec.data
    if allowed is None or len(allowed) == 0 or len(allowed) == vec.shape[dim]:
        return vec.min(dim)[1].item()
    i = None
    for a in allowed:
        if i is None or dim == 0 and vec[a] < vec[i] or dim == 1 and vec[0, a] < vec[0, i] or dim == 2 and vec[0, 0, a] < vec[0, 0, i]:
            i = a
    return i


class Coaching(DAgger):

    def __init__(self, policy, reference, policy_coeff=0.0, p_rollin_ref=NoAnnealing(0)):
        DAgger.__init__(self, policy, reference, p_rollin_ref)
        self.policy_coeff = policy_coeff

    def forward(self, state):
        costs = torch.zeros(self.policy.n_actions)
        self.reference.set_min_costs_to_go(state, costs)
        pred_costs = self.policy.predict_costs(state)
        costs += self.policy_coeff * pred_costs.data
        ref = argmin(costs, state.actions)
        pol = self.policy(state)
        self.objective += self.policy.update(pred_costs, ref, state.actions)
        return ref if self.rollin_ref() else pol


class EpisodeRunner(macarico.Learner):
    REF, LEARN, ACT = 0, 1, 2

    def __init__(self, policy, run_strategy, reference=None, store_ref_costs=False):
        macarico.Learner.__init__(self)
        self.policy = policy
        self.run_strategy = run_strategy
        self.store_ref_costs = store_ref_costs
        self.reference = reference
        self.t = 0
        self.total_loss = 0.0
        self.trajectory = []
        self.limited_actions = []
        self.costs = []
        self.ref_costs = []

    def __call__(self, state):
        a_type = self.run_strategy(self.t)
        pol = self.policy(state) if self.policy is not None else None
        ref_costs_t = None
        if self.store_ref_costs:
            ref_costs_t = torch.zeros(state.n_actions)
            try:
                self.reference.set_min_costs_to_go(state, ref_costs_t)
            except NotImplementedError:
                ref_costs_t = None
        self.ref_costs.append(ref_costs_t)
        if a_type == EpisodeRunner.REF:
            a = self.reference(state)
        elif a_type == EpisodeRunner.LEARN:
            a = pol
        elif isinstance(a_type, tuple) and a_type[0] == EpisodeRunner.ACT:
            a = a_type[1]
        else:
            raise ValueError('run_strategy yielded an invalid choice %s' % a_type)
        assert a in state.actions, 'EpisodeRunner strategy insisting on an illegal action :('
        self.limited_actions.append(list(state.actions))
        self.trajectory.append(a)
        cost = self.policy.predict_costs(state) if self.policy is not None else None
        self.costs.append(cost)
        self.t += 1
        return a


class TiedRandomness(object):

    def __init__(self, rand):
        self.tied = {}
        self.rand = rand

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rand(t)
        return self.tied[t]


def one_step_deviation(T, rollin, rollout, dev_t, dev_a):

    def run(t):
        if t == dev_t:
            return EpisodeRunner.ACT, dev_a
        elif t < dev_t:
            return rollin(t)
        else:
            return rollout(t)
    return run


class LOLS(macarico.LearningAlg):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1

    def __init__(self, policy, reference, loss_fn, p_rollin_ref=NoAnnealing(0), p_rollout_ref=NoAnnealing(0.5), mixture=MIX_PER_ROLL):
        macarico.LearningAlg.__init__(self)
        self.policy = policy
        self.reference = reference
        self.loss_fn = loss_fn()
        self.rollin_ref = stochastic(p_rollin_ref)
        self.rollout_ref = stochastic(p_rollout_ref)
        self.mixture = mixture
        self.rollout = None
        self.true_costs = torch.zeros(self.policy.n_actions)
        self.warned_rollout_ref = False

    def __call__(self, env):
        self.example = env.example
        self.env = env
        n_actions = self.env.n_actions
        loss0, _, _, _, _ = self.run(lambda _: EpisodeRunner.LEARN, True, False)
        _, traj0, limit0, costs0, ref_costs0 = self.run(lambda _: EpisodeRunner.REF if self.rollin_ref() else EpisodeRunner.LEARN, True, True)
        T = len(traj0)
        objective = 0.0
        follow_traj0 = lambda t: (EpisodeRunner.ACT, traj0[t])
        for t, pred_costs in enumerate(costs0):
            true_costs = None
            if self.mixture == LOLS.MIX_PER_ROLL and self.rollout_ref():
                if ref_costs0[t] is None:
                    if not self.warned_rollout_ref:
                        self.warned_rollout_ref = True
                        None
                else:
                    true_costs = ref_costs0[t]
            if true_costs is None:
                true_costs = self.true_costs.zero_()
                rollout = TiedRandomness(self.make_rollout())
                for a in limit0[t]:
                    l, _, _, _, _ = self.run(one_step_deviation(T, follow_traj0, rollout, t, a), False, False)
                    true_costs[a] = float(l)
            true_costs -= true_costs.min()
            objective += self.policy.update(pred_costs, true_costs, limit0[t])
        self.rollin_ref.step()
        self.rollout_ref.step()
        return objective

    def run(self, run_strategy, reset_all, store_ref_costs):
        runner = EpisodeRunner(self.policy, run_strategy, self.reference, store_ref_costs)
        if reset_all:
            self.policy.new_example()
        self.env.run_episode(runner)
        cost = self.loss_fn.evaluate(self.example)
        return cost, runner.trajectory, runner.limited_actions, runner.costs, runner.ref_costs

    def make_rollout(self):
        mk = lambda _: EpisodeRunner.REF if self.rollout_ref() else EpisodeRunner.LEARN
        if self.mixture == LOLS.MIX_PER_ROLL:
            rollout = mk(0)
            return lambda _: rollout
        else:
            return mk


class BanditLOLS(macarico.Learner):
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, EXPLORE_BOOTSTRAP, _EXPLORE_MAX = 0, 1, 2, 3, 4

    def __init__(self, policy, reference=None, p_rollin_ref=NoAnnealing(0), p_rollout_ref=NoAnnealing(0.5), update_method=LEARN_MTR, exploration=EXPLORE_BOLTZMANN, p_explore=NoAnnealing(1.0), mixture=LOLS.MIX_PER_ROLL):
        macarico.Learner.__init__(self)
        if reference is None:
            reference = lambda s: np.random.choice(list(s.actions))
        self.policy = policy
        self.reference = reference
        self.rollin_ref = stochastic(p_rollin_ref)
        self.rollout_ref = stochastic(p_rollout_ref)
        self.update_method = update_method
        self.exploration = exploration
        self.explore = stochastic(p_explore)
        self.mixture = mixture
        assert self.update_method in range(BanditLOLS._LEARN_MAX), 'unknown update_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), 'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        self.dev_t = None
        self.dev_a = None
        self.dev_actions = None
        self.dev_imp_weight = None
        self.dev_costs = None
        self.rollout = None
        self.t = None
        self.disallow = torch.zeros(self.policy.n_actions)
        self.truth = torch.zeros(self.policy.n_actions)

    def forward(self, state):
        if self.t is None:
            self.T = state.horizon()
            self.dev_t = np.random.choice(range(self.T))
            self.t = 0
        a_ref = self.reference(state)
        a_pol = self.policy(state)
        if self.t == self.dev_t:
            a = a_pol
            if self.explore():
                self.dev_costs = self.policy.predict_costs(state)
                self.dev_actions = list(state.actions)[:]
                self.dev_a, self.dev_imp_weight = self.do_exploration(self.dev_costs, self.dev_actions)
                a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0, 0]
        elif self.t < self.dev_t:
            a = a_ref if self.rollin_ref() else a_pol
        else:
            if self.mixture == LOLS.MIX_PER_STATE or self.rollout is None:
                self.rollout = self.rollout_ref()
            a = a_ref if self.rollout else a_pol
        self.t += 1
        return a

    def do_exploration(self, costs, dev_actions):
        if self.exploration == BanditLOLS.EXPLORE_UNIFORM:
            return np.random.choice(list(dev_actions)), len(dev_actions)
        if self.exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            if len(dev_actions) != self.policy.n_actions:
                self.disallow.zero_()
                for i in range(self.policy.n_actions):
                    if i not in dev_actions:
                        self.disallow[i] = 10000000000.0
                costs += Var(self.disallow, requires_grad=False)
            probs = F.softmax(-costs, dim=0)
            a, p = macarico.util.sample_from_probs(probs)
            p = p.data[0]
            if self.exploration == BanditLOLS.EXPLORE_BOLTZMANN_BIASED:
                p = max(p, 0.0001)
            return a, 1 / p
        if self.exploration == BanditLOLS.EXPLORE_BOOTSTRAP:
            assert isinstance(self.policy, macarico.policies.bootstrap.BootstrapPolicy) or isinstance(self.policy, macarico.policies.costeval.CostEvalPolicy)
            probs = costs.get_probs(dev_actions)
            a, p = util.sample_from_np_probs(probs)
            return a, 1 / p
        assert False, 'unknown exploration strategy'

    def get_objective(self, loss):
        loss = float(loss)
        obj = 0.0
        if self.dev_a is not None:
            dev_costs_data = self.dev_costs.data if isinstance(self.dev_costs, Var) else self.dev_costs.data() if isinstance(self.dev_costs, macarico.policies.bootstrap.BootstrapCost) else None
            self.build_truth_vector(loss, self.dev_a, self.dev_imp_weight, dev_costs_data)
            importance_weight = 1.0
            if self.update_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a.data[0, 0]]
                importance_weight = self.dev_imp_weight
            loss_var = self.policy.update(self.dev_costs, self.truth, self.dev_actions)
            loss_var *= importance_weight
            obj = loss_var
        self.explore.step()
        self.rollin_ref.step()
        self.rollout_ref.step()
        self.t, self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs, self.rollout = [None] * 7
        return obj

    def build_truth_vector(self, loss, a, imp_weight, dev_costs_data):
        self.truth.zero_()
        if not isinstance(a, int):
            a = a.data[0, 0]
        if self.update_method == BanditLOLS.LEARN_BIASED:
            self.truth[a] = loss
        elif self.update_method == BanditLOLS.LEARN_IPS:
            self.truth[a] = loss * imp_weight
        elif self.update_method == BanditLOLS.LEARN_DR:
            self.truth += dev_costs_data
            self.truth[a] = dev_costs_data[a] + imp_weight * (loss - dev_costs_data[a])
        elif self.update_method == BanditLOLS.LEARN_MTR:
            self.truth[a] = loss
        elif self.update_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            self.truth[a] = loss - dev_costs_data.min()
        else:
            assert False, self.update_method


class PPO(macarico.Learner):
    """Proximal Policy Optimization"""

    def __init__(self, policy, baseline, epsilon, temperature=1.0, only_one_deviation=False):
        super(PPO, self).__init__()
        self.policy = policy
        self.baseline = baseline
        self.epsilon = epsilon
        self.temperature = temperature
        self.trajectory = []
        self.only_one_deviation = only_one_deviation

    def update(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            dev_t = np.random.choice(range(len(self.trajectory)))
            for t, (a, p_a) in enumerate(self.trajectory):
                if self.only_one_deviation and t != dev_t:
                    continue
                p_a_value = p_a.data
                ratio = p_a * (1 / p_a_value[0])
                ratio_by_adv = (b - loss) * ratio
                lower_bound = dy.constant(1, 1 - self.epsilon)
                clipped_ratio = dy.bmax(ratio, lower_bound)
                upper_bound = dy.constant(1, 1 + self.epsilon)
                clipped_ratio = dy.bmin(clipped_ratio, upper_bound)
                clipped_ratio_by_adv = (b - loss) * clipped_ratio
                increment = dy.bmin(ratio_by_adv, clipped_ratio_by_adv)
                total_loss -= increment
            self.baseline.update(loss)
            if not isinstance(total_loss, float):
                total_loss.forward()
                total_loss.backward()

    def __call__(self, state):
        action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
        self.trajectory.append((action, p_action))
        return action


class EWMA(object):
    """Exponentially weighted moving average."""

    def __init__(self, rate, initial_value=0.0):
        self.rate = rate
        self.value = initial_value

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += self.rate * (x - self.value)

    def __call__(self):
        return self.value


class Reinforce(macarico.Learner):
    """REINFORCE with a scalar baseline function."""

    def __init__(self, policy, baseline=EWMA(0.8)):
        macarico.Learner.__init__(self)
        assert isinstance(policy, StochasticPolicy)
        self.policy = policy
        self.baseline = baseline
        self.trajectory = []

    def get_objective(self, loss):
        if len(self.trajectory) == 0:
            return 0.0
        b = 0 if self.baseline is None else self.baseline()
        total_loss = sum(torch.log(p_a) for p_a in self.trajectory) * (loss - b)
        if self.baseline is not None:
            self.baseline.update(loss)
        self.trajectory = []
        return total_loss

    def forward(self, state):
        action, p_action = self.policy.stochastic(state)
        self.trajectory.append(p_action)
        return action


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


class A2C(macarico.Learner):

    def __init__(self, policy, state_value_fn, value_multiplier=1.0):
        macarico.Learner.__init__(self)
        self.policy = policy
        self.state_value_fn = state_value_fn
        self.trajectory = []
        self.value_multiplier = value_multiplier
        self.loss_fn = nn.SmoothL1Loss()
        self.loss_var = torch.zeros(1)

    def get_objective(self, loss):
        if len(self.trajectory) == 0:
            return
        loss = float(loss)
        loss_var = Varng(self.loss_var + loss)
        total_loss = 0.0
        for p_a, value in self.trajectory:
            v = value.data[0, 0]
            total_loss += (loss - v) * p_a.log()
            total_loss += self.value_multiplier * self.loss_fn(value, loss_var)
        self.trajectory = []
        return total_loss

    def forward(self, state):
        action, p_action = self.policy.stochastic(state)
        value = self.state_value_fn(state)
        self.trajectory.append((p_action, value))
        return action


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
    return [LinearPolicy(features, n_actions, loss_fn=loss_fn, n_layers=n_layers, hidden_dim=hidden_dim) for features in features_bag]


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


class CostEvalPolicy(Policy):

    def __init__(self, reference, policy):
        self.policy = policy
        self.reference = reference
        self.costs = None
        self.n_actions = policy.n_actions
        self.record = [None] * 1000
        self.record_i = 0

    def __call__(self, state):
        if self.costs is None:
            self.costs = torch.zeros(state.n_actions)
        self.costs *= 0
        self.reference.set_min_costs_to_go(state, self.costs)
        p_c = self.policy.predict_costs(state)
        self.record[self.record_i] = sum(abs(self.costs - p_c.data))
        self.record_i = (self.record_i + 1) % len(self.record)
        if np.random.random() < 0.0001 and self.record[-1] is not None:
            None
        return self.policy.greedy(state, pred_costs=p_c)

    def predict_costs(self, state):
        return self.policy.predict_costs(state)

    def forward_partial_complete(self, pred_costs, truth, actions):
        return self.policy.forward_partial_complete(pred_costs, truth, actions)

    def update(self, _):
        pass


class SoftmaxPolicy(macarico.StochasticPolicy):

    def __init__(self, features, n_actions, temperature=1.0):
        macarico.StochasticPolicy.__init__(self)
        self.n_actions = n_actions
        self.features = features
        self.mapping = nn.Linear(features.dim, n_actions)
        self.disallow = torch.zeros(n_actions)
        self.temperature = temperature

    def forward(self, state):
        fts = self.features(state)
        z = self.mapping(fts).squeeze().data
        return util.argmin(z, state.actions)

    def stochastic(self, state):
        z = self.mapping(self.features(state)).squeeze()
        if len(state.actions) != self.n_actions:
            self.disallow.zero_()
            self.disallow += 10000000000.0
            for a in state.actions:
                self.disallow[a] = 0.0
            z += Varng(self.disallow)
        p = F.softmax(-z / self.temperature, dim=0)
        return util.sample_from_probs(p)


def truth_to_vec(truth, tmp_vec):
    if isinstance(truth, torch.FloatTensor):
        return truth
    if isinstance(truth, int) or isinstance(truth, np.int32) or isinstance(truth, np.int64):
        tmp_vec.zero_()
        tmp_vec += 1
        tmp_vec[truth] = 0
        return tmp_vec
    if isinstance(truth, list) or isinstance(truth, set):
        tmp_vec.zero_()
        tmp_vec += 1
        for t in truth:
            tmp_vec[t] = 0
        return tmp_vec
    raise ValueError('invalid argument type for "truth", must be in, list or set; got "%s"' % type(truth))


class CSOAAPolicy(SoftmaxPolicy, CostSensitivePolicy):

    def __init__(self, features, n_actions, loss_fn='huber', temperature=1.0):
        SoftmaxPolicy.__init__(self, features, n_actions, temperature)
        self.set_loss(loss_fn)

    def set_loss(self, loss_fn):
        assert loss_fn in ['squared', 'huber']
        self.loss_fn = nn.MSELoss(size_average=False) if loss_fn == 'squared' else nn.SmoothL1Loss(size_average=False) if loss_fn == 'huber' else None

    def predict_costs(self, state):
        return self.mapping(self.features(state)).squeeze()

    def _compute_loss(self, loss_fn, pred_costs, truth, state_actions):
        if len(state_actions) == self.n_actions:
            return loss_fn(pred_costs, Varng(truth))
        return sum(loss_fn(pred_costs[a], Varng(torch.zeros(1) + truth[a])) for a in state_actions)

    def _update(self, pred_costs, truth, actions=None):
        truth = truth_to_vec(truth, torch.zeros(self.n_actions))
        return self._compute_loss(self.loss_fn, pred_costs, truth, actions)


class WMCPolicy(CSOAAPolicy):

    def __init__(self, features, n_actions, loss_fn='hinge', temperature=1.0):
        CSOAAPolicy.__init__(self, features, n_actions, loss_fn, temperature)

    def set_loss(self, loss_fn):
        assert loss_fn in ['multinomial', 'hinge', 'squared', 'huber']
        if loss_fn == 'hinge':
            l = nn.MultiMarginLoss(size_average=False)
            self.loss_fn = lambda p, t, _: l(p, Varng(torch.LongTensor([t])))
        elif loss_fn == 'multinomial':
            l = nn.NLLLoss(size_average=False)
            self.loss_fn = lambda p, t, _: l(F.log_softmax(p.unsqueeze(0), dim=1), Varng(torch.LongTensor([t])))
        elif loss_fn in ['squared', 'huber']:
            l = (nn.MSELoss if loss_fn == 'squared' else nn.SmoothL1Loss)(size_average=False)
            self.loss_fn = lambda p, t, sa: self._compute_loss(l, p, 1 - truth_to_vec(t, torch.zeros(self.n_actions)), sa)

    def _update(self, pred_costs, truth, actions=None):
        pred_costs = -pred_costs
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list) or isinstance(truth, set):
            return sum(self.loss_fn(pred_costs, a, actions) for a in truth)
        assert isinstance(truth, torch.FloatTensor)
        if len(actions) == 1:
            a = list(actions)[0]
            return self.loss_fn(pred_costs, a, actions)
        full_actions = actions is None or len(actions) == self.n_actions
        truth_sum = truth.sum() if full_actions else sum(truth[a] for a in actions)
        w = truth_sum / (len(actions) - 1) - truth
        w -= w.min()
        return sum(w[a] * self.loss_fn(pred_costs, a, actions) for a in actions if w[a] > 1e-06)


class MultitaskPolicy(macarico.Policy):

    def __init__(self, dispatchers):
        self.dispatchers = dispatchers
        super(MultitaskPolicy, self).__init__()
        all_modules = []
        for d in dispatchers:
            all_modules += d.policy.modules()
        self._dispatcher_module_list = torch.nn.ModuleList(all_modules)

    def forward(self, state):
        for d in self.dispatchers:
            if d.check(state.example):
                return d.policy(state)
        raise Exception('MultitaskLearningAlg dispatching failed on %s' % state.example)

    def state_dict(self):
        return [d.policy.state_dict() for d in self.dispatchers]

    def load_state_dict(self, models):
        assert False


class MultitaskLearningAlg(macarico.LearningAlg):

    def __init__(self, dispatchers):
        self.dispatchers = dispatchers

    def mk_env(self, example):
        for d in self.dispatchers:
            if d.check(example):
                return d.mk_env(example)
        raise Exception('MultitaskLearningAlg.mk_env dispatching failed on %s' % example)

    def policy(self):
        return MultitaskPolicy(self.dispatchers)

    def losses(self):
        return [lambda : MultitaskLoss(self.dispatchers)] + [(lambda i: lambda : MultitaskLoss(self.dispatchers, i))(ii) for ii in range(len(self.dispatchers))]

    def __call__(self, env):
        for d in self.dispatchers:
            if d.check(env.example):
                return d.learn(env)
        raise Exception('MultitaskLearningAlg.__call__ dispatching failed on %s' % ex)


class WAPPolicy(Policy):
    """Linear policy, with weighted all pairs

    Notes:

    This policy can be trained with
    - policy gradient via `policy.stochastic().reinforce(reward)`

    - Cost-sensitive one-against-all linear regression (CSOAA) via
      `policy.forward(state, truth)`

    """

    def __init__(self, features, n_actions):
        self.n_actions = n_actions
        dim = 1 if features is None else features.dim
        n_actions_choose_2 = n_actions * (n_actions - 1) // 2
        self._wap_w = dy_model.add_parameters((n_actions_choose_2, dim))
        self._wap_b = dy_model.add_parameters(n_actions_choose_2)
        self.features = features

    def __call__(self, state):
        return self.greedy(state)

    def sample(self, state):
        return self.stochastic(state)

    def stochastic(self, state, temperature=1):
        return self.stochastic_with_probability(state, temperature)[0]

    def stochastic_with_probability(self, state, temperature=1):
        assert False

    def predict_costs(self, state):
        """Predict costs using the csoaa model accounting for `state.actions`"""
        if self.features is None:
            feats = dy.parameter(self.dy_model.add_parameters(1))
            self.features = lambda _: feats
        else:
            feats = self.features(state)
        wap_w = dy.parameter(self._wap_w)
        wap_b = dy.parameter(self._wap_b)
        return dy.affine_transform([wap_b, wap_w, feats])

    def greedy(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state)
        if isinstance(pred_costs, dy.Expression):
            pred_costs = pred_costs.data
        costs = torch.zeros(self.n_actions)
        k = 0
        for i in xrange(self.n_actions):
            for j in xrange(i):
                costs[i] -= pred_costs[k]
                costs[j] += pred_costs[k]
                k += 1
        if len(state.actions) == self.n_actions:
            return costs.argmin()
        best = None
        for a in state.actions:
            if best is None or costs[a] < costs[best]:
                best = a
        return best

    def forward_partial_complete(self, pred_costs, truth, actions):
        if isinstance(truth, int):
            truth = [truth]
        if isinstance(truth, list):
            truth0 = truth
            truth = torch.ones(self.n_actions)
            for k in truth0:
                truth[k] = 0.0
        obj = 0.0
        k = 0
        if not isinstance(actions, set):
            actions = set(actions)
        for i in xrange(self.n_actions):
            for j in xrange(i):
                weight = abs(truth[i] - truth[j])
                label = -1 if truth[i] > truth[j] else +1
                if weight > 1e-06:
                    l = 1 - label * pred_costs[k]
                    l = 0.5 * (l + dy.abs(l))
                    obj += weight * l
                k += 1
        return obj

    def forward(self, state, truth):
        costs = self.predict_costs(state)
        return self.forward_partial_complete(costs, truth, state.actions)


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    return sum(hand) + (10 if usable_ace(hand) else 0)


class BlackjackFeatures(macarico.DynamicFeatures):

    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, 4)
        view[0, 0, 0] = 1.0
        view[0, 0, 1] = float(sum_hand(state.player))
        view[0, 0, 2] = float(state.dealer[0])
        view[0, 0, 3] = float(usable_ace(state.player))
        return Varng(view)


class CartPoleFeatures(macarico.DynamicFeatures):

    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)

    def _forward(self, state):
        return Var(state.state.view(1, 1, -1), requires_grad=False)


class ConcentrationPOFeatures(macarico.DynamicFeatures):

    def __init__(self):
        super(ConcentrationPOFeatures, self).__init__(8)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        c = util.zeros(self._t, 2)
        c[state.t % 2] = 1
        if state.flipped is None:
            return torch.cat([c, util.zeros(self._t, 6)]).view(1, 1, -1)
        else:
            return torch.cat([c, state.faces[state.card[state.flipped]]]).view(1, 1, -1)


class ConcentrationSmartFeatures(macarico.DynamicFeatures):

    def __init__(self, n_card_types, cheat=False):
        self.cheat = cheat
        self.dim = 2 * n_card_types * n_card_types + n_card_types + 2 + cheat * n_card_types * 2
        self.n_card_types = n_card_types
        super(ConcentrationSmartFeatures, self).__init__(self.dim)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        f = []
        for n in range(2 * self.n_card_types):
            c = util.zeros(self._t, self.n_card_types)
            if n in state.seen:
                c[state.card[n]] = 1
            f.append(c)
        c = util.zeros(self._t, self.n_card_types)
        if state.flipped is not None:
            c[state.card[state.flipped]] = 1
        f.append(c)
        c = util.zeros(self._t, 2)
        c[state.t % 2] = 1
        f.append(c)
        if self.cheat:
            c = util.zeros(self._t, 2 * self.n_card_types)
            c[ConcentrationReference()(state)] = 1
            f.append(c)
        return torch.cat(f).view(1, 1, -1)


class DependencyAttention(macarico.Attention):
    arity = 2

    def __init__(self, features):
        macarico.Attention.__init__(self, features)
        self.oob = self.make_out_of_bounds()

    def _forward(self, state):
        x = self.features(state)
        b = state.b
        i = state.stack[-1] if len(state.stack) > 0 else -1
        return [x[0, b].unsqueeze(0) if b >= 0 and b < state.N else self.oob, x[0, i].unsqueeze(0) if i >= 0 and i < state.N else self.oob]


class GlobalGridFeatures(macarico.DynamicFeatures):

    def __init__(self, width, height):
        macarico.DynamicFeatures.__init__(self, width * height)
        self.width = width
        self.height = height
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        view[0, 0, state.loc[0] * state.example.height + state.loc[1]] = 1
        return Varng(view)

    def __call__(self, state):
        return self.forward(state)


class LocalGridFeatures(macarico.DynamicFeatures):

    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        if not state.is_legal((state.loc[0] - 1, state.loc[1])):
            view[0, 0, 0] = 1.0
        if not state.is_legal((state.loc[0] + 1, state.loc[1])):
            view[0, 0, 1] = 1.0
        if not state.is_legal((state.loc[0], state.loc[1] - 1)):
            view[0, 0, 2] = 1.0
        if not state.is_legal((state.loc[0], state.loc[1] + 1)):
            view[0, 0, 3] = 1.0
        return Varng(view)

    def __call__(self, state):
        return self.forward(state)


class HexFeatures(macarico.DynamicFeatures):

    def __init__(self, board_size=5):
        self.board_size = board_size
        macarico.DynamicFeatures.__init__(self, 3 * self.board_size ** 2)

    def _forward(self, state):
        view = state.state.view(1, 1, 3 * self.board_size ** 2)
        return Varng(view)


class MDPFeatures(macarico.DynamicFeatures):

    def __init__(self, n_states, noise_rate=0):
        macarico.DynamicFeatures.__init__(self, n_states)
        self.n_states = n_states
        self.noise_rate = noise_rate
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        f = util.zeros(self._t.weight, 1, 1, self.n_states)
        if np.random.random() > self.noise_rate:
            f[0, 0, state.s] = 1
        return Varng(f)

    def __call__(self, state):
        return self.forward(state)


class MountainCarFeatures(macarico.DynamicFeatures):

    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 2)

    def _forward(self, state):
        return Var(state.state.view(1, 1, -1))


class PendulumFeatures(macarico.DynamicFeatures):

    def __init__(self):
        macarico.DynamicFeatures.__init__(self, 4)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t.weight, 1, 1, 4)
        view[0, 0, 0] = 1.0
        view[0, 0, 1] = np.cos(state.th)
        view[0, 0, 2] = np.sin(state.th)
        view[0, 0, 3] = state.th_dot
        return Varng(view)


class LocalPOCFeatures(macarico.DynamicFeatures):

    def __init__(self, history_length=1):
        macarico.DynamicFeatures.__init__(self, 10 * history_length)
        self.history_length = history_length
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        view = util.zeros(self._t, 1, 1, 10 * self.history_length)
        for h in range(self.history_length):
            obs = state.obs[max(0, len(state.obs) - h - 1)]
            for i in range(10):
                if obs & i > 0:
                    view[0, 0, h * 10 + i] = 1.0
        return Varng(view)


PASSABLE, SEED, POWER = 0, 1, 2


class GlobalPOCFeatures(macarico.DynamicFeatures):

    def __init__(self, width, height):
        macarico.DynamicFeatures.__init__(self, width * height * 8)
        self.width = width
        self.height = height
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state):
        ghost_positions = set(state.ghost_pos)
        view = util.zeros(self._t, 1, 1, 10 * self.history_length)
        for y in range(self.height):
            for x in range(self.width):
                idx = (x * self.height + y) * 8
                c = 0
                pos = x, y
                if not state.passable(pos):
                    c = 1
                if pos in state.food:
                    c = 3 if state.check(pos, POWER) else 2
                if pos in ghost_positions:
                    c = 5 if state.power_steps == 0 else 7
                elif pos == state.pocman:
                    c = 4 if state.power_steps == 0 else 6
                view[0, 0, idx + c] = 1
        return Varng(view)


class LearnerToAlg(macarico.LearningAlg):

    def __init__(self, learner, policy, loss):
        macarico.LearningAlg.__init__(self)
        self.learner = learner
        self.policy = policy
        self.loss = loss()

    def __call__(self, env):
        env.rewind(self.policy)
        env.run_episode(self.learner)
        loss = self.loss.evaluate(env.example)
        obj = self.learner.get_objective(loss)
        return obj


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NoopLearner,
     lambda: ([], {'policy': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Torch,
     lambda: ([], {'features': _mock_layer(), 'dim': 4, 'layers': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hal3_macarico(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

