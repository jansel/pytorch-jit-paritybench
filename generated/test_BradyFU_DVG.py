import sys
_module = sys.modules[__name__]
del sys
data = _module
generation_dataset = _module
recognition_dataset = _module
misc = _module
util = _module
networks = _module
generator = _module
light_cnn = _module
train_generator = _module
train_lightcnn = _module
val = _module

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


from collections import defaultdict


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import copy


import math


import time


import torch.nn.functional as F


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision.utils as vutils


class _Residual_Block(nn.Module):

    def __init__(self, inc=64, outc=64, groups=1):
        super(_Residual_Block, self).__init__()
        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = None
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


def make_layer(block, num_of_layer, inc=64, outc=64, groups=1):
    if num_of_layer < 1:
        num_of_layer = 1
    layers = []
    layers.append(block(inc=inc, outc=outc, groups=groups))
    for _ in range(1, num_of_layer):
        layers.append(block(inc=outc, outc=outc, groups=groups))
    return nn.Sequential(*layers)


class Encoder(nn.Module):

    def __init__(self, hdim=256):
        super(Encoder, self).__init__()
        self.hdim = hdim
        self.main = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2, bias=False), nn.InstanceNorm2d(32, eps=0.001), nn.LeakyReLU(0.2), nn.AvgPool2d(2), make_layer(_Residual_Block, 1, 32, 64), nn.AvgPool2d(2), make_layer(_Residual_Block, 1, 64, 128), nn.AvgPool2d(2), make_layer(_Residual_Block, 1, 128, 256), nn.AvgPool2d(2), make_layer(_Residual_Block, 1, 256, 512), nn.AvgPool2d(2), make_layer(_Residual_Block, 1, 512, 512))
        self.fc = nn.Linear(512 * 4 * 4, 2 * hdim)

    def forward(self, x):
        z = self.main(x).view(x.size(0), -1)
        z = self.fc(z)
        mu, logvar = torch.split(z, split_size_or_sections=self.hdim, dim=-1)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, hdim=256):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hdim, 512 * 4 * 4), nn.ReLU(True))
        self.main = nn.Sequential(make_layer(_Residual_Block, 1, 512, 512), nn.Upsample(scale_factor=2, mode='nearest'), make_layer(_Residual_Block, 1, 512, 512), nn.Upsample(scale_factor=2, mode='nearest'), make_layer(_Residual_Block, 1, 512, 512), nn.Upsample(scale_factor=2, mode='nearest'), make_layer(_Residual_Block, 1, 512, 256), nn.Upsample(scale_factor=2, mode='nearest'), make_layer(_Residual_Block, 1, 256, 128), nn.Upsample(scale_factor=2, mode='nearest'), make_layer(_Residual_Block, 2, 128, 64), nn.Conv2d(64, 3 + 3, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        x = y.view(z.size(0), -1, 4, 4)
        img = torch.sigmoid(self.main(x))
        return img


class mfm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):

    def __init__(self, block, layers, is_train=False, num_classes=80013):
        super(network_29layers_v2, self).__init__()
        self.is_train = is_train
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)
        if self.is_train:
            self.fc2_ = nn.Linear(256, num_classes, bias=False)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        if self.is_train:
            x = F.dropout(fc, training=self.training)
            out = self.fc2_(x)
            return out, fc
        else:
            return fc


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_Residual_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (group,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (mfm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resblock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_BradyFU_DVG(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

