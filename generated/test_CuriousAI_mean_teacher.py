import sys
_module = sys.modules[__name__]
del sys
unpack_cifar10 = _module
experiments = _module
cifar10_test = _module
imagenet_valid = _module
main = _module
mean_teacher = _module
architectures = _module
cli = _module
data = _module
datasets = _module
losses = _module
ramps = _module
run_context = _module
tests = _module
test_data = _module
utils = _module
cifar10 = _module
preprocess_cifar10 = _module
svhn = _module
test_cifar10 = _module
test_svhn = _module
test_utils = _module
cifar10_final_eval = _module
cifar10_no_augmentation_final_eval = _module
cifar10_supervised_final_eval = _module
cifar10_supervised_no_augmentation_final_eval = _module
svhn_250_vary_consistency_cost = _module
svhn_250_vary_dropout = _module
svhn_250_vary_ema_decay = _module
svhn_250_vary_logit_distance_cost = _module
svhn_250_vary_perturbation = _module
svhn_250_vary_trust = _module
svhn_final_eval = _module
svhn_no_augmentation_final_eval = _module
svhn_supervised_final_eval = _module
svhn_supervised_no_augmentation_final_eval = _module
framework = _module
minibatching = _module
model = _module
nn = _module
string_utils = _module
test_minibatching = _module
weight_norm = _module
train_cifar10 = _module
train_svhn = _module

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


import logging


import torch


import re


import time


import math


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


import torchvision.datasets


import itertools


from torch import nn


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.data.sampler import Sampler


class ShiftConvDownsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels, kernel_size=1, groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNet224x224(nn.Module):

    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 4
        self.downsample_mode = downsample
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels * 8, groups, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(block.out_channels(channels * 8, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(channels * 8, groups), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, block.out_channels(planes, groups), kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(block.out_channels(planes, groups)))
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes, out_channels=block.out_channels(planes, groups))
            else:
                assert False
        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class ResNet32x32(nn.Module):

    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, block.out_channels(planes, groups), kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(block.out_channels(planes, groups)))
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes, out_channels=block.out_channels(planes, groups))
            else:
                assert False
        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class BottleneckBlock(nn.Module):

    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + a)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Shake(Function):

    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1.0 - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1, grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShakeShakeBlock(nn.Module):

    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x
        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)
        ab = shake(a, b, training=self.training)
        if self.downsample is not None:
            residual = self.downsample(x)
        return residual + ab


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ShakeShakeBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShiftConvDownsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_CuriousAI_mean_teacher(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

