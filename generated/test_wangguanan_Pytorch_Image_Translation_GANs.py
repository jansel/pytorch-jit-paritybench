import sys
_module = sys.modules[__name__]
del sys
loaders = _module
main = _module
models = _module
solver = _module
loaders = _module
models = _module
solver = _module
loaders = _module
models = _module
solver = _module
tools = _module

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


import torch.utils.data as data


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import torch


import torch.nn as nn


import torch.optim as optim


from torchvision.utils import save_image


import itertools


from torch.utils import data


from torchvision.datasets import ImageFolder


import random


import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.model(x) + x


class Generator(nn.Module):
    """Generator with Down sampling, Several ResBlocks and Up sampling.
       Down/Up Samplings are used for less computation.
    """

    def __init__(self, class_num, conv_dim, layer_num):
        super(Generator, self).__init__()
        self.class_num = class_num
        layers = []
        layers.append(nn.Conv2d(in_channels=3 + class_num, out_channels=conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        current_dims = conv_dim
        for i in xrange(2):
            layers.append(nn.Conv2d(current_dims, current_dims * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims *= 2
        for i in xrange(layer_num):
            layers.append(ResidualBlock(current_dims, current_dims))
        for i in xrange(2):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims // 2
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat([1, 1, x.size(2), x.size(3)])
        x = torch.cat([x, c], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator with PatchGAN"""

    def __init__(self, image_size, conv_dim, layer_num, class_num):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        current_dim = conv_dim
        for i in xrange(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            current_dim *= 2
        self.model = nn.Sequential(*layers)
        kernel_size = int(image_size / 2 ** layer_num)
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(current_dim, class_num, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        out_src = self.conv_src(x)
        out_cls = self.conv_cls(x)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'class_num': 4, 'conv_dim': 4, 'layer_num': 1}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 1, 1])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNetDown,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wangguanan_Pytorch_Image_Translation_GANs(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

