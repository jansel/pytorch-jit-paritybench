import sys
_module = sys.modules[__name__]
del sys
config = _module
test = _module
test_config = _module
train = _module
test = _module
train = _module
dataloader = _module
mscocoMulti = _module
label_transform = _module
networks = _module
globalNet = _module
network = _module
refineNet = _module
resnet = _module
utils = _module
evaluation = _module
imutils = _module
logger = _module
misc = _module
osutils = _module
transforms = _module

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


import time


import matplotlib.pyplot as plt


import torch


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torchvision.datasets as datasets


import numpy as np


import random


import math


import torch.utils.data as data


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from random import randint


import scipy.misc


import scipy.io


class globalNet(nn.Module):

    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)
        return global_fms, global_outs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class refineNet(nn.Module):

    def __init__(self, lateral_channel, out_shape, num_class):
        super(refineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        out = torch.cat(refine_fms, dim=1)
        out = self.final_predict(out)
        return out


class CPN(nn.Module):

    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)
        return global_outs, refine_out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4, x3, x2, x1]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_GengDavid_pytorch_cpn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

