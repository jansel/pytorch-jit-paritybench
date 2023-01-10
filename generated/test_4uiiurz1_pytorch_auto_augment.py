import sys
_module = sys.modules[__name__]
del sys
auto_augment = _module
test = _module
train = _module
utils = _module
wide_resnet = _module

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


import time


import math


from collections import OrderedDict


import random


import warnings


import numpy as np


import matplotlib.pyplot as plt


import pandas as pd


from sklearn.model_selection import train_test_split


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


import torchvision


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import torchvision.transforms as transforms


import torchvision.datasets as datasets


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.shortcut = conv1x1(inplanes, planes, stride)
            self.use_conv1x1 = True
        else:
            self.use_conv1x1 = False

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        if self.use_conv1x1:
            shortcut = self.shortcut(out)
        else:
            shortcut = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += shortcut
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, width, num_classes=10, dropout=0.3):
        super(WideResNet, self).__init__()
        layer = (depth - 4) // 6
        self.inplanes = 16
        self.conv = conv3x3(3, 16)
        self.layer1 = self._make_layer(16 * width, layer, dropout)
        self.layer2 = self._make_layer(32 * width, layer, dropout, stride=2)
        self.layer3 = self._make_layer(64 * width, layer, dropout, stride=2)
        self.bn = nn.BatchNorm2d(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * width, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes, blocks, dropout, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(self.inplanes, planes, dropout, stride if i == 0 else 1))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_4uiiurz1_pytorch_auto_augment(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

