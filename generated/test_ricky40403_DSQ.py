import sys
_module = sys.modules[__name__]
del sys
DSQConv = _module
DSQLinear = _module
train = _module

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


import random


import time


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


class RoundWithGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = x / delta + 0.5
        return x.round() * 2 - 1

    @staticmethod
    def backward(ctx, g):
        return g


class DSQConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, momentum=0.1, num_bit=8, QInput=True, bSetQ=True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2 ** self.num_bit - 1
        self.is_quan = bSetQ
        self.momentum = momentum
        if self.is_quan:
            self.uW = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            self.lW = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())
            if self.bias is not None:
                self.uB = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lB = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))
                self.alphaB = nn.Parameter(data=torch.tensor(0.2).float())
            if self.quan_input:
                self.uA = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lA = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data]))
                self.register_buffer('running_lA', torch.tensor([self.lA.data]))
                self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x

    def phi_function(self, x, mi, alpha, delta):
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]), alpha)
        s = 1 / (1 - alpha)
        k = (2 / alpha - 1).log() * (1 / delta)
        x = ((x - mi) * k).tanh() * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)
        return x

    def dequantize(self, x, lower_bound, delta, interval):
        x = ((x + 1) / 2 + interval) * delta + lower_bound
        return x

    def forward(self, x):
        if self.is_quan:
            if self.training:
                cur_running_lw = self.running_lw.mul(1 - self.momentum).add(self.momentum * self.lW)
                cur_running_uw = self.running_uw.mul(1 - self.momentum).add(self.momentum * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw
            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta = (cur_max - cur_min) / self.bit_range
            interval = (Qweight - cur_min) // delta
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)
            Qbias = self.bias
            if self.bias is not None:
                if self.training:
                    cur_running_lB = self.running_lB.mul(1 - self.momentum).add(self.momentum * self.lB)
                    cur_running_uB = self.running_uB.mul(1 - self.momentum).add(self.momentum * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB
                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta = (cur_max - cur_min) / self.bit_range
                interval = (Qbias - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)
            Qactivation = x
            if self.quan_input:
                if self.training:
                    cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
                    cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA
                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta = (cur_max - cur_min) / self.bit_range
                interval = (Qactivation - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class DSQLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, num_bit=4, QInput=True, bSetQ=True):
        super(DSQLinear, self).__init__(in_features, out_features, bias=bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2 ** self.num_bit - 1
        self.is_quan = bSetQ
        self.momentum = momentum
        if self.is_quan:
            self.uW = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            self.lW = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())
            if self.bias is not None:
                self.uB = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lB = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))
                self.alphaB = nn.Parameter(data=torch.tensor(0.2).float())
            if self.quan_input:
                self.uA = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lA = nn.Parameter(data=torch.tensor(-1 * 2 ** 32).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data]))
                self.register_buffer('running_lA', torch.tensor([self.lA.data]))
                self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x

    def phi_function(self, x, mi, alpha, delta):
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]), alpha)
        s = 1 / (1 - alpha)
        k = (2 / alpha - 1).log() * (1 / delta)
        x = ((x - mi) * k).tanh() * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)
        return x

    def dequantize(self, x, lower_bound, delta, interval):
        x = ((x + 1) / 2 + interval) * delta + lower_bound
        return x

    def forward(self, x):
        if self.is_quan:
            if self.training:
                cur_running_lw = self.running_lw.mul(1 - self.momentum).add(self.momentum * self.lW)
                cur_running_uw = self.running_uw.mul(1 - self.momentum).add(self.momentum * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw
            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta = (cur_max - cur_min) / self.bit_range
            interval = (Qweight - cur_min) // delta
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)
            Qbias = self.bias
            if self.bias is not None:
                if self.training:
                    cur_running_lB = self.running_lB.mul(1 - self.momentum).add(self.momentum * self.lB)
                    cur_running_uB = self.running_uB.mul(1 - self.momentum).add(self.momentum * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB
                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta = (cur_max - cur_min) / self.bit_range
                interval = (Qbias - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)
            Qactivation = x
            if self.quan_input:
                if self.training:
                    cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
                    cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA
                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta = (cur_max - cur_min) / self.bit_range
                interval = (Qactivation - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
            output = F.linear(Qactivation, Qweight, Qbias)
        else:
            output = F.linear(x, self.weight, self.bias)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DSQConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ricky40403_DSQ(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

