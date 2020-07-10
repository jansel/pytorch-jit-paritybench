import sys
_module = sys.modules[__name__]
del sys
AttModel = _module
data_load = _module
eval = _module
hyperparams = _module
modules = _module
prepro = _module
train = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import *


import numpy as np


import random


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import torch.optim as optim


import time


class embedding(nn.Module):

    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        """Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        """
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[(0), :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = self._backend.Embedding.apply(inputs, self.lookup_table, self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * self.num_units ** 0.5
        return outputs


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-08):
        """Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        """
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):
        """Point-wise feed forward net.

        Args:
          in_channels: a number of channels of inputs
          num_units: A list of two integers.
        """
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units
        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0], 'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1], 'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            self.conv1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        if self.conv:
            inputs = inputs.permute(0, 2, 1)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += inputs
        if self.conv:
            outputs = self.normalization(outputs.permute(0, 2, 1))
        else:
            outputs = self.normalization(outputs)
        return outputs


class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
        """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

        Args:
            epsilon: Smoothing rate.
        """
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return (1 - self.epsilon) * inputs + self.epsilon / K


class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False):
        """Applies multihead attention.

        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        """
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))
        outputs = outputs / K_.size()[-1] ** 0.5
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)
        padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1))
        condition = key_masks.eq(0.0).float()
        outputs = padding * condition + outputs * (1.0 - condition)
        if self.causality:
            diag_vals = torch.ones(*outputs[(0), :, :].size())
            tril = torch.tril(diag_vals, diagonal=0)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))
            padding = Variable(torch.ones(*masks.size()) * (-2 ** 32 + 1))
            condition = masks.eq(0.0).float()
            outputs = padding * condition + outputs * (1.0 - condition)
        outputs = F.softmax(outputs, dim=-1)
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])
        outputs = outputs * query_masks
        outputs = self.output_dropout(outputs)
        outputs = torch.bmm(outputs, V_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        outputs += queries
        outputs = self.normalization(outputs)
        return outputs


class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):
        """Sinusoidal Positional_Encoding.

        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
        """
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        N, T = inputs.size()[0:2]
        position_ind = Variable(torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).long())
        position_enc = torch.Tensor([[(pos / np.power(10000, 2.0 * i / self.num_units)) for i in range(self.num_units)] for pos in range(T)])
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        lookup_table = Variable(position_enc)
        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)), lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1
        outputs = self._backend.Embedding.apply(position_ind, lookup_table, padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * self.num_units ** 0.5
        return outputs


class AttModel(nn.Module):

    def __init__(self, hp_, enc_voc, dec_voc):
        """Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        """
        super(AttModel, self).__init__()
        self.hp = hp_
        self.enc_voc = enc_voc
        self.dec_voc = dec_voc
        self.enc_emb = embedding(self.enc_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(num_units=self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units, [4 * self.hp.hidden_units, self.hp.hidden_units]))
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(num_units=self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units, [4 * self.hp.hidden_units, self.hp.hidden_units]))
        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):
        self.decoder_inputs = torch.cat([Variable(torch.ones(y[:, :1].size()) * 2).long(), y[:, :-1]], dim=-1)
        self.enc = self.enc_emb(x)
        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x)
        else:
            self.enc += self.enc_positional_encoding(Variable(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long()))
        self.enc = self.enc_dropout(self.enc)
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        self.dec = self.dec_emb(self.decoder_inputs)
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs)
        else:
            self.dec += self.dec_positional_encoding(Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long()))
        self.dec = self.dec_dropout(self.dec)
        for i in range(self.hp.num_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)
        self.logits = self.logits_layer(self.dec)
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc)
        _, self.preds = torch.max(self.logits, -1)
        self.istarget = (1.0 - y.eq(0.0).float()).view(-1)
        self.acc = torch.sum(self.preds.eq(y).float().view(-1) * self.istarget) / torch.sum(self.istarget)
        self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc)
        self.y_onehot = Variable(self.y_onehot.scatter_(1, y.view(-1, 1).data, 1))
        self.y_smoothed = self.label_smoothing(self.y_onehot)
        self.loss = -torch.sum(self.y_smoothed * torch.log(self.probs), dim=-1)
        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)
        return self.mean_loss, self.preds, self.acc


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (label_smoothing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (layer_normalization,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_leviswind_pytorch_transformer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

