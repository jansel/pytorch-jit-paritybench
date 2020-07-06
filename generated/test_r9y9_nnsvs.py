import sys
_module = sys.modules[__name__]
del sys
conf = _module
data_prep = _module
nnsvs = _module
base = _module
bin = _module
fit_scaler = _module
generate = _module
prepare_features = _module
preprocess_normalize = _module
synthesis = _module
train = _module
data = _module
data_source = _module
dsp = _module
frontend = _module
ja = _module
gen = _module
hts = _module
logger = _module
model = _module
multistream = _module
util = _module
version = _module
setup = _module
tests = _module

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


import torch


from torch import nn


from torch.nn import functional as F


from abc import ABC


from abc import abstractmethod


import numpy as np


from scipy.io import wavfile


from torch.utils import data as data_utils


from torch import optim


from torch.backends import cudnn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils import weight_norm


import random


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(dilation), WNConv1d(dim, dim, kernel_size=3, dilation=dilation), nn.LeakyReLU(0.2), WNConv1d(dim, dim, kernel_size=1))
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Conv1dResnet(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, dropout=0.0):
        super().__init__()
        model = [nn.ReflectionPad1d(3), WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0)]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(hidden_dim, out_dim, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x, lengths=None):
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class FeedForwardNet(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super(FeedForwardNet, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
        return self.last_linear(h)


class LSTMRNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, out_dim)

    def forward(self, sequence, lengths):
        sequence = pack_padded_sequence(sequence, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForwardNet,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LSTMRNN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
]

class Test_r9y9_nnsvs(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

