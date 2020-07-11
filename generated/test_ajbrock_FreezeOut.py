import sys
_module = sys.modules[__name__]
del sys
WRN = _module
densenet = _module
train = _module
utils = _module
vgg = _module

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


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import numpy as np


from torch.autograd import Variable


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


import torchvision.models as models


import logging


from torch.autograd import Variable as V


import time


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate, layer_index):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        if self.active:
            return out
        else:
            return out.detach()


class Layer(nn.Module):

    def __init__(self, n_in, n_out, layer_index):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.layer_index = layer_index
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = F.relu(self.bn1(self.conv1(x)))
        if self.active:
            return out
        else:
            return out.detach()


scale_fn = {'linear': lambda x: x, 'squared': lambda x: x ** 2, 'cubic': lambda x: x ** 3}


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate, layer_index):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        if self.active:
            return out
        else:
            return out.detach()


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate, layer_index):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.layer_index = layer_index
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = torch.cat((x, out), 1)
        if self.active:
            return out
        else:
            return out.detach()


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels, layer_index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = F.avg_pool2d(out, 2)
        if self.active:
            return out
        else:
            return out.detach()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bottleneck,
     lambda: ([], {'nChannels': 4, 'growthRate': 4, 'layer_index': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Layer,
     lambda: ([], {'n_in': 4, 'n_out': 4, 'layer_index': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4, 'layer_index': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4, 'layer_index': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ajbrock_FreezeOut(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

