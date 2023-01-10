import sys
_module = sys.modules[__name__]
del sys
datagen = _module
ptflops = _module
flops_counter = _module
sample = _module
setup = _module
loss = _module
net = _module
get_state_dict = _module
shufflenetv2 = _module
summary = _module
example = _module
model_hook = _module
model_summary = _module
module_mac = _module
summary_tree = _module
test = _module
shufflenetv2 = _module
test = _module
test_without_crop = _module
httpClient = _module
shufflenetv2 = _module
shufflenetv2_to_caffe = _module
test = _module
get_topk = _module
load = _module
load_test_image = _module
test_model = _module
train = _module
train_affectnet = _module

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


import numpy as np


import torchvision.models as models


import torch.nn.functional as F


from torch.autograd import Variable


import math


import torch.nn.init as init


from collections import OrderedDict


from torch.nn import init


import time


import queue


import pandas as pd


import torch.optim as optim


from torch.optim import lr_scheduler


import torchvision


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import scipy.io as scio


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


from torchvision import utils


class FIIQALoss(nn.Module):

    def __init__(self):
        super(AGLoss, self).__init__()

    def forward(self, fiiqa_preds, fiiqa_targets):
        """Compute loss (fiiqa_preds, fiiqa_targets) .

        Args:
          fiiqa_preds: (tensor) predicted fiiqa, sized [batch_size,100].
          fiiqa_targets: (tensor) target fiiqa, sized [batch_size,].

        loss:
          (tensor) loss = SmoothL1Loss(fiiqa_preds, fiiqa_targets)
        """
        fiiqa_prob = F.softmax(fiiqa_preds, dim=1)
        fiiqa_expect = torch.sum(Variable(torch.arange(0, 200)).float() * fiiqa_prob, 1)
        fiiqa_loss = F.smooth_l1_loss(fiiqa_expect, fiiqa_targets.float())
        return fiiqa_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.training = False


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class AGNet(nn.Module):

    def __init__(self):
        super(AGNet, self).__init__()
        self.backbone = ResNet18()
        self.age_head = self._make_head()
        self.linear1 = nn.Linear(256, 200)

    def _make_head(self):
        return nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        N = x.size(0)
        y = self.backbone(x)
        y = F.avg_pool2d(y, 5)
        y_age = self.age_head(y)
        y_age = self.linear1(y_age.view(N, -1))
        return y_age


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2
        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True), nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm2d(oup_inc), nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
        else:
            self.banch1 = nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
            self.banch2 = nn.Sequential(nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True), nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm2d(oup_inc), nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))
        return channel_shuffle(out, 2)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class ShuffleNetV2(nn.Module):

    def __init__(self, input_size, n_class=200, width_mult=1):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 192]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError('width_mult = {} is not supported'.format(width_mult))
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


class SoftmaxWrapper(nn.Module):

    def __init__(self, net):
        super(SoftmaxWrapper, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return F.softmax(y, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AGNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetV2,
     lambda: ([], {'input_size': 32}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SoftmaxWrapper,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yangyuke001_FIIQA_PyTorch(_paritybench_base):
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

