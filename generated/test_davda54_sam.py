import sys
_module = sys.modules[__name__]
del sys
cifar = _module
smooth_cross_entropy = _module
wide_res_net = _module
train = _module
bypass_bn = _module
cutout = _module
initialize = _module
loading_bar = _module
log = _module
step_lr = _module
sam = _module

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


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn.modules.batchnorm import _BatchNorm


import random


class BasicUnit(nn.Module):

    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([('0_normalization', nn.BatchNorm2d(channels)), ('1_activation', nn.ReLU(inplace=True)), ('2_convolution', nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)), ('3_normalization', nn.BatchNorm2d(channels)), ('4_activation', nn.ReLU(inplace=True)), ('5_dropout', nn.Dropout(dropout, inplace=True)), ('6_convolution', nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False))]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([('0_normalization', nn.BatchNorm2d(in_channels)), ('1_activation', nn.ReLU(inplace=True))]))
        self.block = nn.Sequential(OrderedDict([('0_convolution', nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)), ('1_normalization', nn.BatchNorm2d(out_channels)), ('2_activation', nn.ReLU(inplace=True)), ('3_dropout', nn.Dropout(dropout, inplace=True)), ('4_convolution', nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False))]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(DownsampleUnit(in_channels, out_channels, stride, dropout), *(BasicUnit(out_channels, dropout) for _ in range(depth)))

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):

    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
        super(WideResNet, self).__init__()
        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)
        self.f = nn.Sequential(OrderedDict([('0_convolution', nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)), ('1_block', Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)), ('2_block', Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)), ('3_block', Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)), ('4_normalization', nn.BatchNorm2d(self.filters[3])), ('5_activation', nn.ReLU(inplace=True)), ('6_pooling', nn.AvgPool2d(kernel_size=8)), ('7_flattening', nn.Flatten()), ('8_classification', nn.Linear(in_features=self.filters[3], out_features=labels))]))
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicUnit,
     lambda: ([], {'channels': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'depth': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownsampleUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_davda54_sam(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

