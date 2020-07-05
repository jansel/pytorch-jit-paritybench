import sys
_module = sys.modules[__name__]
del sys
lstms = _module
container = _module
lstm = _module
normalize = _module
setup = _module
test_capacity = _module
test_container = _module
test_correctness = _module
test_normalize = _module
test_speed = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import torch as th


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor as T


from torch.nn import Parameter as P


from torch.autograd import Variable as V


from torch.optim import Adam


from torch.optim import SGD


from time import time


class MultiLayerLSTM(nn.Module):
    """
    MultiLayer LSTM of any type.
    
    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size, layer_type, layer_sizes=(64, 64), *args, **kwargs):
        super(MultiLayerLSTM, self).__init__()
        rnn = layer_type
        layers = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = rnn(*args, input_size=prev_size, hidden_size=size, **kwargs)
            layers.append(layer)
            prev_size = size
        if 'dropout' in kwargs:
            del kwargs['dropout']
        layer = rnn(*args, input_size=prev_size, hidden_size=layer_sizes[-1], dropout=0.0, **kwargs)
        layers.append(layer)
        self.layers = layers
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def create_hiddens(self, bsz=1):
        hiddens = []
        for l in self.layers:
            std = math.sqrt(2.0 / (l.input_size + l.hidden_size))
            hiddens.append([V(T(1, bsz, l.hidden_size).normal_(0, std)), V(T(1, bsz, l.hidden_size).normal_(0, std))])
        return hiddens

    def sample_mask(self):
        for l in self.layers:
            l.sample_mask()

    def forward(self, x, hiddens):
        new_hiddens = []
        for l, h in zip(self.layers, hiddens):
            None
            x, new_h = l(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens


class SlowLSTM(nn.Module):
    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass


class LSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf

    Special args:
    
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta']
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)
        preact = self.i2h(x) + self.h2h(h)
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)
        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)
        h_t = th.mul(o_t, c_t.tanh())
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-06):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).expand_as(x)) / th.sqrt(th.var(x, 1).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class BradburyLayerNorm(nn.Module):
    """
    Layer Norm, according to:
    https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139
    """

    def __init__(self, features, eps=1e-06):
        super(BradburyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(th.ones(features))
        self.beta = nn.Parameter(th.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class BaLayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    This implementation mimicks the original torch implementation at:
    https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-05):
        super(BaLayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.epsilon = epsilon
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        mean = th.mean(x, 1).expand_as(x)
        center = x - mean
        std = th.sqrt(th.mean(th.square(center), 1)).expand_as(x)
        output = center / (std + self.epsilon)
        if self.learnable:
            output = self.alpha * output + self.beta
        return output.view(size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaLayerNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (BradburyLayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_seba_1511_lstms_pth(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

