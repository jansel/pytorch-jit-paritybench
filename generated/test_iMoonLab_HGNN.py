import sys
_module = sys.modules[__name__]
del sys
config = _module
datasets = _module
data_helper = _module
visual_data = _module
HGNN = _module
models = _module
layers = _module
train = _module
hypergraph_utils = _module

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


from torch import nn


import torch.nn.functional as F


import math


import torch


import torch.nn as nn


from torch.nn.parameter import Parameter


import time


import copy


import torch.optim as optim


class HGNN_conv(nn.Module):

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN(nn.Module):

    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HGNN_fc(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):

    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):

    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (HGNN,
     lambda: ([], {'in_ch': 4, 'n_class': 4, 'n_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGNN_classifier,
     lambda: ([], {'n_hid': 4, 'n_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGNN_conv,
     lambda: ([], {'in_ft': 4, 'out_ft': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGNN_embedding,
     lambda: ([], {'in_ch': 4, 'n_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HGNN_fc,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_iMoonLab_HGNN(_paritybench_base):
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

