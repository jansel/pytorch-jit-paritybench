import sys
_module = sys.modules[__name__]
del sys
dataset = _module
kor_char_parser = _module
main = _module
tcn = _module

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


import random


import numpy as np


from torch.utils.data import Dataset


import re


import torch


from torch.autograd import Variable


from torch import nn


from torch import optim


from torch.utils.data import DataLoader


import torch.nn.functional as F


import math


import torch.nn as nn


from torch.nn.utils import weight_norm


GPU_NUM = True


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """

    def __init__(self, dims, k_size, v_size, seq_len=None):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2, 1))
        mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
        mask = torch.from_numpy(mask)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits, dim=1) / self.sqrt_k
        read = torch.bmm(probs, values)
        return minibatch + read


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        if self.downsample is not None:
            nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))

    def forward(self, x):
        net = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(net + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, max_length=200, attention=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            if attention == True:
                layers += [AttentionBlock(max_length, max_length, max_length)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table """
    position_enc = np.array([([(pos / np.power(10000, 2 * (j // 2) / d_pos_vec)) for j in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec)) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class TCN(nn.Module):

    def __init__(self, embedding_dim: int, max_length: int, channel=200, level=3, kernel_size=3, dropout=0.2, emb_dropout=0.0, tied_weights=False, attention=False):
        super(TCN, self).__init__()
        self.channel = channel
        self.channels = [channel] * level
        self.embedding_dim = embedding_dim
        self.character_size = 252
        self.max_length = max_length
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim, padding_idx=0)
        self.pe = nn.Embedding(self.max_length, self.embedding_dim, padding_idx=0)
        self.pe.weight.data.copy_(position_encoding_init(self.max_length, self.embedding_dim))
        self.pe.weight.requires_grad = False
        self.tcn = TemporalConvNet(embedding_dim, self.channels, kernel_size, dropout=dropout, max_length=max_length, attention=attention)

    def forward(self, inputs, lens):
        data_in_torch = Variable(torch.from_numpy(np.array(inputs)).long())
        len_in_torch = Variable(torch.from_numpy(np.array(lens)).long())
        if GPU_NUM:
            data_in_torch = data_in_torch
            len_in_torch = len_in_torch
        embeds = self.embeddings(data_in_torch)
        pe = self.pe(len_in_torch)
        embeds += pe
        output = self.tcn(embeds.transpose(1, 2)).transpose(1, 2)
        return output.contiguous()


class TNT(nn.Module):

    def __init__(self, embedding_dim: int, max_length: int, channel_size=200, T_size=16, level=3, attention=False):
        super(TNT, self).__init__()
        self.tcn = TCN(embedding_dim, max_length, channel=channel_size, level=level, attention=attention)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.output_dim = 1
        self.fc1 = nn.Linear(self.max_length * channel_size, T_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(T_size, 5)
        self.init_weights()

    def init_weights(self):
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def forward(self, inputs, lens):
        sent = self.tcn(inputs, lens)
        sent = sent.view(sent.size(0), -1)
        net = self.act1(self.fc1(sent))
        out = self.fc2(net)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionBlock,
     lambda: ([], {'dims': 4, 'k_size': 4, 'v_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TCN,
     lambda: ([], {'embedding_dim': 4, 'max_length': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (TNT,
     lambda: ([], {'embedding_dim': 4, 'max_length': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (TemporalConvNet,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_flrngel_TCN_with_attention(_paritybench_base):
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

