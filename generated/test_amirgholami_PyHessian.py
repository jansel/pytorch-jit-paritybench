import sys
_module = sys.modules[__name__]
del sys
density_plot = _module
example_pyhessian_analysis = _module
resnet = _module
pyhessian = _module
hessian = _module
utils = _module
training = _module
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


import math


from copy import deepcopy


import logging


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, residual_not, batch_norm_not, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, residual_not, batch_norm_not, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.batch_norm_not:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)
        return out


ALPHA_ = 1


class ResNet(nn.Module):

    def __init__(self, depth, residual_not=True, batch_norm_not=True, base_channel=16, num_classes=10):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = BasicBlock
        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = self.base_channel * ALPHA_
        self.conv1 = nn.Conv2d(3, self.base_channel * ALPHA_, kernel_size=3, padding=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(self.base_channel * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.base_channel * ALPHA_, n, self.residual_not, self.batch_norm_not)
        self.layer2 = self._make_layer(block, self.base_channel * 2 * ALPHA_, n, self.residual_not, self.batch_norm_not, stride=2)
        self.layer3 = self._make_layer(block, self.base_channel * 4 * ALPHA_, n, self.residual_not, self.batch_norm_not, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.base_channel * 4 * ALPHA_ * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, residual_not, batch_norm_not, stride=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion) and residual_not:
            if batch_norm_not:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
            else:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, residual_not, batch_norm_not, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual_not, batch_norm_not))
        return layers

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        if self.batch_norm_not:
            x = self.bn1(x)
        x = self.relu(x)
        output_list.append(x.view(x.size(0), -1))
        for layer in self.layer1:
            x = layer(x)
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x)
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x)
            output_list.append(x.view(x.size(0), -1))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output_list.append(x.view(x.size(0), -1))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'residual_not': 4, 'batch_norm_not': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_amirgholami_PyHessian(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

