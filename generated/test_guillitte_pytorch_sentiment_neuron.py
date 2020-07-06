import sys
_module = sys.modules[__name__]
del sys
convert_to_cpu = _module
lm = _module
load_from_numpy = _module
models = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from torch.autograd import Variable


from torch import optim


import torch.nn.functional as F


import torch.nn as nn


import numpy as np


import time


import math


class mLSTM(nn.Module):

    def __init__(self, data_size, hidden_size, n_layers=1):
        super(mLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.data_size = data_size
        self.n_layers = n_layers
        input_size = data_size + hidden_size
        self.wx = nn.Linear(data_size, 4 * hidden_size, bias=False)
        self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.wmx = nn.Linear(data_size, hidden_size, bias=False)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, data, last_hidden):
        hx, cx = last_hidden
        m = self.wmx(data) * self.wmh(hx)
        gates = self.wx(data) + self.wh(m)
        i, f, o, u = gates.chunk(4, 1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        u = F.tanh(u)
        o = F.sigmoid(o)
        cy = f * cx + i * u
        hy = o * F.tanh(cy)
        return hy, cy


class StackedLSTM(nn.Module):

    def __init__(self, cell, num_layers, input_size, rnn_size, output_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.h2o = nn.Linear(rnn_size, output_size)
        self.layers = []
        for i in range(num_layers):
            layer = cell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            if i == 0:
                input = h_1_i
            else:
                input = input + h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        output = self.h2o(input)
        return (h_1, c_1), output

    def state0(self, batch_size):
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size), requires_grad=False)
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size), requires_grad=False)
        return h_0, c_0

