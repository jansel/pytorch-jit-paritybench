import sys
_module = sys.modules[__name__]
del sys
core = _module
test = _module
train = _module
DatasetLoader = _module
TargetDatasetLoader = _module
datasets = _module
samplers = _module
main = _module
evaluate = _module
saver = _module
utils = _module
DGFANet = _module
models = _module

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


from collections import OrderedDict


import torchvision.utils as vutils


import torch


import torch.optim as optim


from torch import nn


from torch.nn import DataParallel


import numpy as np


import torch.nn.functional as F


import itertools


import random


import torch.autograd as autograd


from copy import deepcopy


from itertools import permutations


from itertools import combinations


from torchvision import transforms


from torchvision import utils


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from collections import defaultdict


from torch.utils.data.sampler import Sampler


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


from torch.nn import init


import scipy.io as scio


import math


import torchvision.models as models


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)


class inconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(conv3x3(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()
        self.downconv = nn.Sequential(conv3x3(in_channels, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True), conv3x3(128, 196), nn.BatchNorm2d(196), nn.ReLU(inplace=True), conv3x3(196, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        x = self.downconv(x)
        return x


class DepthEstmator(nn.Module):

    def __init__(self, in_channels=384, out_channels=1):
        super(DepthEstmator, self).__init__()
        self.conv = nn.Sequential(conv3x3(in_channels, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True), conv3x3(128, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True), conv3x3(64, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class FeatExtractor(nn.Module):

    def __init__(self, in_channels=6):
        super(FeatExtractor, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = Downconv(64, 128)
        self.down2 = Downconv(128, 128)
        self.down3 = Downconv(128, 128)

    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)
        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)
        return catfeat, dx4


def conv_block(index, in_channels, out_channels, K_SIZE=3, stride=1, padding=1, momentum=0.1, pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(OrderedDict([('conv' + str(index), nn.Conv2d(in_channels, out_channels, K_SIZE, stride=stride, padding=padding)), ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum, affine=True)), ('relu' + str(index), nn.ReLU(inplace=True)), ('pool' + str(index), nn.MaxPool2d(2))]))
    else:
        conv = nn.Sequential(OrderedDict([('conv' + str(index), nn.Conv2d(in_channels, out_channels, K_SIZE, padding=padding)), ('bn' + str(index), nn.BatchNorm2d(out_channels, momentum=momentum, affine=True)), ('relu' + str(index), nn.ReLU(inplace=True))]))
    return conv


class FeatEmbedder(nn.Module):

    def __init__(self, in_channels=128, momentum=0.1):
        super(FeatEmbedder, self).__init__()
        self.momentum = momentum
        self.features = nn.Sequential(conv_block(0, in_channels=in_channels, out_channels=128, momentum=self.momentum, pooling=True), conv_block(1, in_channels=128, out_channels=256, momentum=self.momentum, pooling=True), conv_block(2, in_channels=256, out_channels=512, momentum=self.momentum, pooling=False), nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('fc', nn.Linear(512, 1))

    def forward(self, x, params=None):
        if params == None:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            out = F.conv2d(x, params['features.0.conv0.weight'], params['features.0.conv0.bias'], padding=1)
            out = F.batch_norm(out, params['features.0.bn0.running_mean'], params['features.0.bn0.running_var'], params['features.0.bn0.weight'], params['features.0.bn0.bias'], momentum=self.momentum, training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)
            out = F.conv2d(out, params['features.1.conv1.weight'], params['features.1.conv1.bias'], padding=1)
            out = F.batch_norm(out, params['features.1.bn1.running_mean'], params['features.1.bn1.running_var'], params['features.1.bn1.weight'], params['features.1.bn1.bias'], momentum=self.momentum, training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)
            out = F.conv2d(out, params['features.2.conv2.weight'], params['features.2.conv2.bias'], padding=1)
            out = F.batch_norm(out, params['features.2.bn2.running_mean'], params['features.2.bn2.running_var'], params['features.2.bn2.weight'], params['features.2.bn2.bias'], momentum=self.momentum, training=True)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out = F.linear(out, params['fc.weight'], params['fc.bias'])
        return out

    def cloned_state_dict(self):
        cloned_state_dict = {key: val.clone() for key, val in self.state_dict().items()}
        return cloned_state_dict


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DepthEstmator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Downconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatEmbedder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     False),
    (inconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_rshaojimmy_RFMetaFAS(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

