import sys
_module = sys.modules[__name__]
del sys
dataset = _module
dataset_360D = _module
exporters = _module
image = _module
file_utils = _module
infer = _module
models = _module
modules = _module
resnet360 = _module
spherical = _module
cartesian = _module
derivatives = _module
grid = _module
weights = _module
supervision = _module
direct = _module
photometric = _module
smoothness = _module
splatting = _module
ssim = _module
test = _module
train_lr = _module
train_sv = _module
train_tc = _module
train_ud = _module
utils = _module
checkpoint = _module
framework = _module
init = _module
meters = _module
opt = _module
visualization = _module

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


import numpy as np


import torch


from torch.utils.data import Dataset


from torchvision import transforms


import warnings


import numpy


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import functools


import math


import torchvision


import random


import torch.optim as optim


from torch.optim import Optimizer


class AddCoords360(nn.Module):

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords360, self).__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.float32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.float32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)
        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)
        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv360(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, with_r, in_channels, out_channels, kernel_size, *args, **kwargs):
        super(CoordConv360, self).__init__()
        self.addcoords = AddCoords360(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret


class Identity(nn.Module):

    def forward(self, x):
        return x


def create_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)


def create_conv(in_size, out_size, conv_type, padding=1, stride=1, kernel_size=3, width=512):
    if conv_type == 'standard':
        return nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, padding=padding, stride=stride)
    elif conv_type == 'coord':
        return CoordConv360(x_dim=width / 2.0, y_dim=width, with_r=False, kernel_size=kernel_size, stride=stride, in_channels=in_size, out_channels=out_size, padding=padding)


def create_normalization(out_size, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(out_size)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(out_size // 4, out_size)
    elif norm_type == 'none':
        return Identity()


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_type, conv_type, activation, width):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [create_conv(dim, dim, conv_type, width=width), create_normalization(dim, norm_type), create_activation(activation)]
        conv_block += [create_conv(dim, dim, conv_type, width=width), create_normalization(dim, norm_type)]
        self.block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)
        return out


class ResNet360(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, in_channels=3, out_channels=1, depth=5, wf=32, conv_type='coord', padding='kernel', norm_type='none', activation='elu', up_mode='upconv', down_mode='downconv', width=512, use_dropout=False, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert depth >= 0
        super(ResNet360, self).__init__()
        model = [create_conv(in_channels, wf, conv_type, kernel_size=7, padding=3, stride=1, width=width), create_normalization(wf, norm_type), create_activation(activation)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [create_conv(wf * mult, wf * mult * 2, conv_type, kernel_size=3, stride=2, padding=1, width=width // (i + 1)), create_normalization(wf * mult * 2, norm_type), create_activation(activation)]
        mult = 2 ** n_downsampling
        for i in range(depth):
            model += [ResnetBlock(wf * mult, activation=activation, norm_type=norm_type, conv_type=conv_type, width=width // 2 ** n_downsampling)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(wf * mult, int(wf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), create_normalization(int(wf * mult / 2), norm_type), create_activation(activation)]
        model += [create_conv(wf, out_channels, conv_type, kernel_size=7, padding=3, width=width)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CoordConv360,
     lambda: ([], {'x_dim': 4, 'y_dim': 4, 'with_r': 4, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_VCL3D_SphericalViewSynthesis(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

