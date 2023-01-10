import sys
_module = sys.modules[__name__]
del sys
arch = _module
discriminators = _module
generators = _module
ops = _module
main = _module
model = _module
test = _module
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


from torch import nn


import torch


from torch.nn import functional as F


import functools


from torch.nn import init


import torch.nn as nn


import itertools


from torch.autograd import Variable


import torchvision.datasets as dsets


import torchvision.transforms as transforms


from torch.optim import lr_scheduler


import torchvision


import copy


import numpy as np


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), norm_layer(out_dim), nn.LeakyReLU(0.2, True))


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, norm_layer=norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, norm_layer=norm_layer, padding=1, bias=use_bias)]
        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)


def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1), conv_norm_relu(dim, dim, kernel_size=3, norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding, bias=bias), norm_layer(out_dim), nn.ReLU(True))


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        res_model = [nn.ReflectionPad2d(3), conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias), conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias), conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]
        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]
        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'dim': 4, 'norm_layer': _mock_layer, 'use_dropout': 0.5, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResnetGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_arnab39_cycleGAN_PyTorch(_paritybench_base):
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

