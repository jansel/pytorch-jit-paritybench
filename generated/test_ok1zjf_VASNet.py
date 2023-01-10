import sys
_module = sys.modules[__name__]
del sys
config = _module
cpd_auto = _module
cpd_nonlin = _module
create_split = _module
knapsack = _module
layer_norm = _module
main = _module
sys_utils = _module
vasnet_model = _module
vsum_tools = _module

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


from torch.autograd import Variable


import torch


import torch.nn as nn


from torchvision import transforms


import numpy as np


import time


import random


import torch.nn.init as init


from torch.nn.modules.module import _addindent


import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()
        self.apperture = apperture
        self.ignore_itself = ignore_itself
        self.m = input_size
        self.output_size = output_size
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)
        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        n = x.shape[0]
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1, 0))
        if self.ignore_itself:
            logits[torch.eye(n).byte()] = -float('Inf')
        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float('Inf')
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1, 0), weights).transpose(1, 0)
        y = self.output_linear(y)
        return y, att_weights_


class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()
        self.m = 1024
        self.hidden_size = 1024
        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)

    def forward(self, x, seq_len):
        m = x.shape[2]
        x = x.view(-1, m)
        y, att_weights_ = self.att(x)
        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ok1zjf_VASNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

