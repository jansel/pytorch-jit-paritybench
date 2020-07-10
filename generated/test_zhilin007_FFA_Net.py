import sys
_module = sys.modules[__name__]
del sys
net = _module
data_utils = _module
main = _module
metrics = _module
FFA = _module
PerceptualLoss = _module
models = _module
option = _module
test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torchvision.transforms as tfs


from torchvision.transforms import functional as FF


import numpy as np


import torch


import random


from torch.utils.data import DataLoader


from torchvision.utils import make_grid


import torchvision


import time


import math


from torch.backends import cudnn


from torch import optim


import warnings


from torch import nn


import torchvision.utils as vutils


from torchvision.models import vgg16


from math import exp


import torch.nn.functional as F


from torch.autograd import Variable


from torchvision.transforms import ToPILImage


import torch.nn as nn


class PALayer(nn.Module):

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):

    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):

    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class FFA(nn.Module):

    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True), nn.Sigmoid()])
        self.palayer = PALayer(self.dim)
        post_precess = [conv(self.dim, self.dim, kernel_size), conv(self.dim, 3, kernel_size)]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, (None), (None)]
        out = w[:, (0), :] * res1 + w[:, (1), :] * res2 + w[:, (2), :] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


class LossNetwork(torch.nn.Module):

    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3'}

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return sum(loss) / len(loss)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CALayer,
     lambda: ([], {'channel': 18}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
    (FFA,
     lambda: ([], {'gps': 3, 'blocks': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PALayer,
     lambda: ([], {'channel': 18}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
]

class Test_zhilin007_FFA_Net(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

