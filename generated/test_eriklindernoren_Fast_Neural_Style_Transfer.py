import sys
_module = sys.modules[__name__]
del sys
models = _module
test_on_image = _module
test_on_video = _module
train = _module
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


from collections import namedtuple


from torchvision import models


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torchvision.utils import save_image


import random


import numpy as np


from torch.optim import Adam


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


class VGG16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True), ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False))

    def forward(self, x):
        return self.block(x) + x


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(ConvBlock(3, 32, kernel_size=9, stride=1), ConvBlock(32, 64, kernel_size=3, stride=2), ConvBlock(64, 128, kernel_size=3, stride=2), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), ConvBlock(128, 64, kernel_size=3, upsample=True), ConvBlock(64, 32, kernel_size=3, upsample=True), ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False))

    def forward(self, x):
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_eriklindernoren_Fast_Neural_Style_Transfer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

