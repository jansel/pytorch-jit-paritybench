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


import math


from typing import List


from typing import Tuple


from typing import Type


import torch as th


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Parameter


from typing import Iterable


from torch.autograd import Variable as V


from torch.optim import Adam


from torch.optim import SGD


from time import time


from torch import Tensor as T


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

    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0, dropout_method: str='pytorch'):
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
        self.mask = th.bernoulli(th.empty(1, self.hidden_size).fill_(keep))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]) ->Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
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


class MultiLayerLSTM(nn.Module):
    """
    MultiLayer LSTM of any type.

    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size: int, layer_type: Type[LSTM], layer_sizes: List[int]=(64, 64), *args, **kwargs):
        super(MultiLayerLSTM, self).__init__()
        rnn = layer_type
        self.layers: List[LSTM] = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = rnn(*args, input_size=prev_size, hidden_size=size, **kwargs)
            self.layers.append(layer)
            prev_size = size
        if 'dropout' in kwargs:
            del kwargs['dropout']
        if len(layer_sizes) > 0:
            layer = rnn(*args, input_size=prev_size, hidden_size=layer_sizes[-1], dropout=0.0, **kwargs)
            self.layers.append(layer)
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(self.layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def create_hiddens(self, batch_size: int=1) ->List[Tuple[th.Tensor, th.Tensor]]:
        hiddens: List[Tuple[th.Tensor, th.Tensor]] = []
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.input_size + layer.hidden_size))
            hiddens.append((th.empty(1, batch_size, layer.hidden_size).normal_(0, std), th.empty(1, batch_size, layer.hidden_size).normal_(0, std)))
        return hiddens

    def sample_mask(self):
        for layer in self.layers:
            layer.sample_mask()

    def forward(self, x: th.Tensor, hiddens: Tuple[th.Tensor, th.Tensor]) ->Tuple[th.Tensor, List[Tuple[th.Tensor, th.Tensor]]]:
        new_hiddens: List[Tuple[th.Tensor, th.Tensor]] = []
        for layer, h in zip(self.layers, hiddens):
            x, new_h = layer(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens


class SlowLSTM(nn.Module):
    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.w_xi = Parameter(th.empty(hidden_size, input_size))
        self.w_xf = Parameter(th.empty(hidden_size, input_size))
        self.w_xo = Parameter(th.empty(hidden_size, input_size))
        self.w_xc = Parameter(th.empty(hidden_size, input_size))
        self.w_hi = Parameter(th.empty(hidden_size, hidden_size))
        self.w_hf = Parameter(th.empty(hidden_size, hidden_size))
        self.w_ho = Parameter(th.empty(hidden_size, hidden_size))
        self.w_hc = Parameter(th.empty(hidden_size, hidden_size))
        self.b_i = th.empty(hidden_size).fill_(0)
        self.b_f = th.empty(hidden_size).fill_(0)
        self.b_o = th.empty(hidden_size).fill_(0)
        self.b_c = th.empty(hidden_size).fill_(0)
        if bias:
            self.b_i = Parameter(self.b_i)
            self.b_f = Parameter(self.b_f)
            self.b_o = Parameter(self.b_o)
            self.b_c = Parameter(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]) ->Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
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


class GalLSTM(LSTM):
    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, *args, **kwargs):
        super(GalLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'gal'
        self.sample_mask()


class MoonLSTM(LSTM):
    """
    Implementation of Moon & al.:
    'RNNDrop: A Novel Dropout for RNNs in ASR'
    https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf
    """

    def __init__(self, *args, **kwargs):
        super(MoonLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'moon'
        self.sample_mask()


class SemeniutaLSTM(LSTM):
    """
    Implementation of Semeniuta & al.:
    'Recurrent Dropout without Memory Loss'
    https://arxiv.org/pdf/1603.05118.pdf
    """

    def __init__(self, *args, **kwargs):
        super(SemeniutaLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'semeniuta'


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size: int, learnable: bool=True, epsilon: float=1e-06):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = th.empty(1, input_size).fill_(0)
        self.beta = th.empty(1, input_size).fill_(0)
        self.epsilon = epsilon
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: th.Tensor) ->th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).unsqueeze(1)) / th.sqrt(th.var(x, 1).unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class LayerNormLSTM(LSTM):
    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0, dropout_method: str='pytorch', ln_preact: bool=True, learnable: bool=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, dropout=dropout, dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4 * hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4 * hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]) ->Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h
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
        c_t = self.ln_cell(c_t)
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


class LayerNormGalLSTM(LayerNormLSTM):
    """
    Mixes GalLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormGalLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'gal'
        self.sample_mask()


class LayerNormMoonLSTM(LayerNormLSTM):
    """
    Mixes MoonLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormMoonLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'moon'
        self.sample_mask()


class LayerNormSemeniutaLSTM(LayerNormLSTM):
    """
    Mixes SemeniutaLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super(LayerNormSemeniutaLSTM, self).__init__(*args, **kwargs)
        self.dropout_method = 'semeniuta'


class BradburyLayerNorm(nn.Module):
    """
    Layer Norm, according to:
    https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139
    """

    def __init__(self, features: Iterable[int], eps: float=1e-06):
        super(BradburyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(th.ones(features))
        self.beta = nn.Parameter(th.zeros(features))
        self.eps = eps

    def forward(self, x: th.Tensor) ->th.Tensor:
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

    def __init__(self, input_size: int, learnable: bool=True, epsilon: float=1e-05):
        super(BaLayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.epsilon = epsilon
        self.alpha = th.empty(1, input_size).fill_(0)
        self.beta = th.empty(1, input_size).fill_(0)
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def forward(self, x: th.Tensor) ->th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        mean = th.mean(x, 1).unsqueeze(1)
        center = x - mean
        std = th.sqrt(th.mean(th.square(center), 1)).unsqueeze(1)
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

