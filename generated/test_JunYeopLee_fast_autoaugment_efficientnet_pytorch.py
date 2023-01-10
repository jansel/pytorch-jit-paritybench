import sys
_module = sys.modules[__name__]
del sys
eval = _module
fast_auto_augment = _module
networks = _module
basenet = _module
efficientnet = _module
efficientnet_cifar10 = _module
resnet_cifar10 = _module
train = _module
transforms = _module
utils = _module

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


import copy


import time


import random


import torchvision.transforms as transforms


from torch.utils.data import Subset


from sklearn.model_selection import StratifiedShuffleSplit


import math


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


import numpy as np


from abc import ABC


from abc import abstractmethod


import collections


import torchvision


import torchvision.models as models


class BaseNet(nn.Module):

    def __init__(self, backbone, args):
        super(BaseNet, self).__init__()
        self.first = nn.Sequential(*list(backbone.children())[:1])
        self.after = nn.Sequential(*list(backbone.children())[1:-1])
        self.fc = list(backbone.children())[-1]
        self.img_size = 224, 224

    def forward(self, x):
        f = self.first(x)
        x = self.after(f)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, f


class Swish(nn.Module):
    """ Swish activation function, s(x) = x * sigmoid(x) """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


def get_activation_fn(activation):
    if activation == 'swish':
        return Swish
    elif activation == 'relu':
        return nn.ReLU
    else:
        raise Exception('Unkown activation %s' % activation)


class ConvBlock(nn.Module):
    """ Conv + BatchNorm + Activation """

    def __init__(self, in_channel, out_channel, kernel_size, padding=0, stride=1, activation='swish'):
        super().__init__()
        self.fw = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=False), nn.BatchNorm2d(out_channel), get_activation_fn(activation)())

    def forward(self, x):
        return self.fw(x)


class DepthwiseConvBlock(nn.Module):
    """ DepthwiseConv2D + BatchNorm + Activation """

    def __init__(self, in_channel, kernel_size, padding=0, stride=1, activation='swish'):
        super().__init__()
        self.fw = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, stride=stride, groups=in_channel, bias=False), nn.BatchNorm2d(in_channel), get_activation_fn(activation)())

    def forward(self, x):
        return self.fw(x)


class SEBlock(nn.Module):
    """ Squeeze and Excitation Block """

    def __init__(self, in_channel, se_ratio=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        inter_channel = in_channel // se_ratio
        self.reduce = nn.Sequential(nn.Conv2d(in_channel, inter_channel, kernel_size=1, padding=0, stride=1), nn.ReLU())
        self.expand = nn.Sequential(nn.Conv2d(inter_channel, in_channel, kernel_size=1, padding=0, stride=1), nn.Sigmoid())

    def forward(self, x):
        s = self.global_avgpool(x)
        s = self.reduce(s)
        s = self.expand(s)
        return x * s


class MBConv(nn.Module):
    """ Inverted residual block """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, expand_ratio=1, activation='swish', use_seblock=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.use_seblock = use_seblock
        if expand_ratio != 1:
            self.expand = ConvBlock(in_channel, in_channel * expand_ratio, 1, activation=activation)
        self.dw_conv = DepthwiseConvBlock(in_channel * expand_ratio, kernel_size, padding=(kernel_size - 1) // 2, stride=stride, activation=activation)
        if use_seblock:
            self.seblock = SEBlock(in_channel * expand_ratio)
        self.pw_conv = ConvBlock(in_channel * expand_ratio, out_channel, 1, activation=activation)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            x = self.expand(inputs)
        else:
            x = inputs
        x = self.dw_conv(x)
        if self.use_seblock:
            x = self.seblock(x)
        x = self.pw_conv(x)
        if self.in_channel == self.out_channel and self.stride == 1:
            x = x + inputs
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride), nn.BatchNorm2d(out_channel))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel))
        if self.in_channel != self.out_channel or self.stride != 1:
            self.down = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channel))

    def forward(self, b):
        t = self.conv1(b)
        t = self.relu(t)
        t = self.conv2(t)
        if self.in_channel != self.out_channel or self.stride != 1:
            b = self.down(b)
        t += b
        t = self.relu(t)
        return t


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale
        self.stem = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(*[ResidualBlock(16, 16, 1) for _ in range(2 * scale)])
        self.layer2 = nn.Sequential(*[ResidualBlock(in_channel=16 if i == 0 else 32, out_channel=32, stride=2 if i == 0 else 1) for i in range(2 * scale)])
        self.layer3 = nn.Sequential(*[ResidualBlock(in_channel=32 if i == 0 else 64, out_channel=64, stride=2 if i == 0 else 1) for i in range(2 * scale)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s = self.stem(x)
        x = self.layer1(s)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, s


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseConvBlock,
     lambda: ([], {'in_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConv,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     False),
    (Net,
     lambda: ([], {'args': _mock_config(scale=1)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_JunYeopLee_fast_autoaugment_efficientnet_pytorch(_paritybench_base):
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

