import sys
_module = sys.modules[__name__]
del sys
module = _module
train = _module
utils = _module
vgg = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import numpy as np


import torchvision


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import random


import time


import logging


class LegoConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, n_split, n_lego):
        super(LegoConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.n_split = in_channels, out_channels, kernel_size, n_split
        self.basic_channels = in_channels // self.n_split
        self.n_lego = int(self.out_channels * n_lego)
        self.lego = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_lego, self.basic_channels, self.kernel_size, self.kernel_size)))
        self.aux_coefficients = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))
        self.aux_combination = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))

    def forward(self, x):
        self.proxy_combination = torch.zeros(self.aux_combination.size())
        self.proxy_combination.scatter_(2, self.aux_combination.argmax(dim=2, keepdim=True), 1)
        self.proxy_combination.requires_grad = True
        out = 0
        for i in range(self.n_split):
            lego_feature = F.conv2d(x[:, i * self.basic_channels:(i + 1) * self.basic_channels], self.lego, padding=self.kernel_size // 2)
            kernel = self.aux_coefficients[i] * self.proxy_combination[i]
            out = out + F.conv2d(lego_feature, kernel)
        return out

    def copy_grad(self, balance_weight):
        self.aux_combination.grad = self.proxy_combination.grad
        idxs = self.aux_combination.argmax(dim=2).view(-1).cpu().numpy()
        unique, count = np.unique(idxs, return_counts=True)
        unique, count = np.unique(count, return_counts=True)
        avg_freq = self.n_split * self.out_channels / self.n_lego
        max_freq = 0
        min_freq = 100
        for i in range(self.n_lego):
            i_freq = (idxs == i).sum().item()
            max_freq = max(max_freq, i_freq)
            min_freq = min(min_freq, i_freq)
            if i_freq >= np.floor(avg_freq) and i_freq <= np.ceil(avg_freq):
                continue
            if i_freq < np.floor(avg_freq):
                self.aux_combination.grad[:, :, (i)] = self.aux_combination.grad[:, :, (i)] - balance_weight * (np.floor(avg_freq) - i_freq)
            if i_freq > np.ceil(avg_freq):
                self.aux_combination.grad[:, :, (i)] = self.aux_combination.grad[:, :, (i)] + balance_weight * (i_freq - np.ceil(avg_freq))


class lego_vgg16(nn.Module):

    def __init__(self, vgg_name, n_split, n_lego, n_classes):
        super(lego_vgg16, self).__init__()
        self.n_split, self.n_lego, self.n_classes = n_split, n_lego, n_classes
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if i == 0:
                layers += [nn.Conv2d(in_channels, x, 3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
                continue
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [LegoConv2d(in_channels, x, 3, self.n_split, self.n_lego), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def copy_grad(self, balance_weight):
        for layer in self.features.children():
            if isinstance(layer, LegoConv2d):
                layer.copy_grad(balance_weight)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LegoConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'n_split': 4, 'n_lego': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_huawei_noah_LegoNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

