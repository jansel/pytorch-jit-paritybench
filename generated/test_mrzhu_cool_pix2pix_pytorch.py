import sys
_module = sys.modules[__name__]
del sys
data = _module
dataset = _module
networks = _module
test = _module
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


import random


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


from math import log10


import torch.optim as optim


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


class Down(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(out_ch), nn.ReLU(True))

    def forward(self, x):
        x = self.down(x)
        return x


class Inconv(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0, bias=use_bias), norm_layer(out_ch), nn.ReLU(True))

    def forward(self, x):
        x = self.inconv(x)
        return x


class Outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, x):
        x = self.outconv(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(out_ch), nn.ReLU(True))

    def forward(self, x):
        x = self.up(x)
        return x


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)
        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)
        self.outc = Outconv(ngf, output_nc)

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])
        return self.outc(out['u2'])


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'norm_layer': _mock_layer, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (Inconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'norm_layer': _mock_layer, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'norm_layer': _mock_layer, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mrzhu_cool_pix2pix_pytorch(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

