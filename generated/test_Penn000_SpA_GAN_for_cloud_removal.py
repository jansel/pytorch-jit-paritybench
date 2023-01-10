import sys
_module = sys.modules[__name__]
del sys
data_manager = _module
demo = _module
eval = _module
log_report = _module
dis = _module
SPANet = _module
layers = _module
models_utils = _module
predict = _module
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


import numpy as np


from torch.utils import data


import matplotlib.pyplot as plt


import time


import torch


from torch.autograd import Variable


import torch.nn as nn


from collections import OrderedDict


from torch import nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.backends import cudnn


from torch import optim


from torch.nn import functional as F


class CBR(nn.Module):

    def __init__(self, ch0, ch1, bn=True, sample='down', activation=nn.ReLU(True), dropout=False):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        if sample == 'down':
            self.c = nn.Conv2d(ch0, ch1, 4, 2, 1)
        else:
            self.c = nn.ConvTranspose2d(ch0, ch1, 4, 2, 1)
        if bn:
            self.batchnorm = nn.BatchNorm2d(ch1, affine=True)
        if dropout:
            self.Dropout = nn.Dropout()

    def forward(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = self.Dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class _Discriminator(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.in_ch]
        x_1 = x[:, self.in_ch:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):

    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))
        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class irnn_layer(nn.Module):

    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return top_up, top_right, top_down, top_left


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class SAM(nn.Module):

    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask


class SPANet(nn.Module):

    def __init__(self):
        super(SPANet, self).__init__()
        self.conv_in = nn.Sequential(conv3x3(3, 32), nn.ReLU(True))
        self.SAM1 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.conv_out = nn.Sequential(conv3x3(32, 3))

    def forward(self, x):
        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
        Attention1 = self.SAM1(out)
        out = F.relu(self.res_block4(out) * Attention1 + out)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)
        Attention2 = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        Attention3 = self.SAM1(out)
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        Attention4 = self.SAM1(out)
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)
        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
        out = self.conv_out(out)
        return Attention4, out


class Generator(nn.Module):

    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gen = nn.Sequential(OrderedDict([('gen', SPANet())]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)


class UpSamplePixelShuffle(nn.Module):

    def __init__(self, in_ch, out_ch, up_scale=2, activation=nn.ReLU(True)):
        super().__init__()
        self.activation = activation
        self.c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch * up_scale * up_scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps = nn.PixelShuffle(up_scale)

    def forward(self, x):
        h = self.c(x)
        h = self.ps(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBR,
     lambda: ([], {'ch0': 4, 'ch1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Generator,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SAM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPANet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UpSamplePixelShuffle,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (irnn_layer,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Penn000_SpA_GAN_for_cloud_removal(_paritybench_base):
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

