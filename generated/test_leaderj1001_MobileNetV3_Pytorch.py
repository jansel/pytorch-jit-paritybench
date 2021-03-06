import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
preprocess = _module

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


import torch.optim as optim


import time


import torch.nn.functional as F


import torchvision.transforms as transforms


import torchvision.datasets as datasets


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3.0, self.inplace) / 6.0
        return out * x


class SqueezeBlock(nn.Module):

    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(nn.Linear(exp_size, exp_size // divide), nn.ReLU(inplace=True), nn.Linear(exp_size // divide, exp_size), h_sigmoid())

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2
        self.use_connect = stride == 1 and in_channels == out_channels
        if self.nonLinear == 'RE':
            activation = nn.ReLU
        else:
            activation = h_swish
        self.conv = nn.Sequential(nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(exp_size), activation(inplace=True))
        self.depth_conv = nn.Sequential(nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size), nn.BatchNorm2d(exp_size))
        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)
        self.point_conv = nn.Sequential(nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), activation(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)
        if self.SE:
            out = self.squeeze_block(out)
        out = self.point_conv(out)
        if self.use_connect:
            return x + out
        else:
            return out


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class MobileNetV3(nn.Module):

    def __init__(self, model_mode='LARGE', num_classes=1000, multiplier=1.0, dropout_rate=0.0):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        if model_mode == 'LARGE':
            layers = [[16, 16, 3, 1, 'RE', False, 16], [16, 24, 3, 2, 'RE', False, 64], [24, 24, 3, 1, 'RE', False, 72], [24, 40, 5, 2, 'RE', True, 72], [40, 40, 5, 1, 'RE', True, 120], [40, 40, 5, 1, 'RE', True, 120], [40, 80, 3, 2, 'HS', False, 240], [80, 80, 3, 1, 'HS', False, 200], [80, 80, 3, 1, 'HS', False, 184], [80, 80, 3, 1, 'HS', False, 184], [80, 112, 3, 1, 'HS', True, 480], [112, 112, 3, 1, 'HS', True, 672], [112, 160, 5, 1, 'HS', True, 672], [160, 160, 5, 2, 'HS', True, 672], [160, 160, 5, 1, 'HS', True, 960]]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=True))
            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)
            out_conv1_in = _make_divisible(160 * multiplier)
            out_conv1_out = _make_divisible(960 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1), nn.BatchNorm2d(out_conv1_out), h_swish(inplace=True))
            out_conv2_in = _make_divisible(960 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1))
        elif model_mode == 'SMALL':
            layers = [[16, 16, 3, 2, 'RE', True, 16], [16, 24, 3, 2, 'RE', False, 72], [24, 24, 3, 1, 'RE', False, 88], [24, 40, 5, 2, 'RE', True, 96], [40, 40, 5, 1, 'RE', True, 240], [40, 40, 5, 1, 'RE', True, 240], [40, 48, 5, 1, 'HS', True, 120], [48, 48, 5, 1, 'HS', True, 144], [48, 96, 5, 2, 'HS', True, 288], [96, 96, 5, 1, 'HS', True, 576], [96, 96, 5, 1, 'HS', True, 576]]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=True))
            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)
            out_conv1_in = _make_divisible(96 * multiplier)
            out_conv1_out = _make_divisible(576 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1), SqueezeBlock(out_conv1_out), nn.BatchNorm2d(out_conv1_out), h_swish(inplace=True))
            out_conv2_in = _make_divisible(576 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1))
        self.apply(_weights_init)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = self.out_conv2(out).view(batch, -1)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MobileBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernal_size': 4, 'stride': 1, 'nonLinear': 4, 'SE': 4, 'exp_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (MobileNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SqueezeBlock,
     lambda: ([], {'exp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_leaderj1001_MobileNetV3_Pytorch(_paritybench_base):
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

