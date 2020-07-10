import sys
_module = sys.modules[__name__]
del sys
master = _module
experiment = _module
model = _module
utils = _module
experiment = _module
model = _module
utils = _module
experiment = _module
model = _module
utils = _module
experiment = _module
model = _module
utils = _module
RTransformer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torch.optim as optim


import time


import math


import warnings


from torch import nn


import torch.nn.functional as F


from scipy.io import loadmat


import torch


import numpy as np


from torchvision import datasets


from torchvision import transforms


from collections import Counter


from random import randint


import copy


class LocalRNN(nn.Module):

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)
        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx)
        self.zeros = torch.zeros((self.ksize - 1, input_dim))

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x)
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, (-1), :]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LayerNorm(nn.Module):
    """Construct a layernorm module."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LocalRNNLayer(nn.Module):
    """Encoder is made up of attconv and feed forward (defined below)"""

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.connection(x, self.local_rnn)
        return x


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MHPooling(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        """Take in model size and number of heads."""
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        attn_shape = 1, 3000, 3000
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)

    def forward(self, x):
        """Implements Figure 2"""
        nbatches, seq_len, d_model = x.shape
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (x, x, x))]
        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len], dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Block(nn.Module):
    """
    One Block
    """

    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        self.layers = clones(LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.connections[0](x, self.pooling)
        x = self.connections[1](x, self.feed_forward)
        return x


class RTransformer(nn.Module):
    """
    The overal model
    """

    def __init__(self, d_model, rnn_type, ksize, n_level, n, h, dropout):
        super(RTransformer, self).__init__()
        N = n
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, dropout)
        layers = []
        for i in range(n_level):
            layers.append(Block(d_model, d_model, rnn_type, ksize, N=N, h=h, dropout=dropout))
        self.forward_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_net(x)
        return x


class RT(nn.Module):

    def __init__(self, input_size, output_size, h, rnn_type, ksize, n_level, n, dropout=0.2, emb_dropout=0.2, tied_weights=False):
        super(RT, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.rt = RTransformer(input_size, rnn_type, ksize, n_level, n, h, dropout)
        self.decoder = nn.Linear(input_size, output_size)
        if tied_weights:
            self.decoder.weight = self.encoder.weight
            None
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.rt(emb)
        y = self.decoder(y)
        return y.contiguous()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'rnn_type': 4, 'ksize': 4, 'N': 4, 'h': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalRNN,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'rnn_type': 4, 'ksize': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LocalRNNLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'rnn_type': 4, 'ksize': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MHPooling,
     lambda: ([], {'d_model': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RT,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'h': 4, 'rnn_type': 4, 'ksize': 4, 'n_level': 4, 'n': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (RTransformer,
     lambda: ([], {'d_model': 4, 'rnn_type': 4, 'ksize': 4, 'n_level': 4, 'n': 4, 'h': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
]

class Test_DSE_MSU_R_transformer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

