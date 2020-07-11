import sys
_module = sys.modules[__name__]
del sys
datasets = _module
pascalvoc = _module
discriminators = _module
discriminator = _module
evaluate = _module
generators = _module
deeplabv2 = _module
train = _module
utils = _module
helpers = _module
lr_scheduling = _module
mattopng = _module
metrics = _module
transforms = _module
validate = _module

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


from torchvision.transforms import Compose


import torch.nn as nn


import torchvision.models as models


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.autograd import Variable


import torchvision.transforms as transforms


from torchvision.transforms import ToTensor


import math


import random


from collections import OrderedDict


from torchvision import transforms


from functools import reduce


import torch.optim as optim


import torch.utils.model_zoo as model_zoo


class Dis(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """

    def __init__(self, in_channels, negative_slope=0.2):
        super(Dis, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self.conv1 = nn.Conv2d(in_channels=self._in_channels, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope, inplace=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = x[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)
        x = x[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)
        x = x[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)
        x = x[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)
        x = x[:, :, :-1, :-1]
        return x


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
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


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], NoLabels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.upsample_bilinear(x, scale_factor=2)[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)[:, :, :-1, :-1]
        x = F.upsample_bilinear(x, scale_factor=2)[:, :, :-1, :-1]
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier_Module,
     lambda: ([], {'dilation_series': [4, 4], 'padding_series': [4, 4], 'NoLabels': 4}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (Dis,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ms_sharma_Adversarial_Semisupervised_Semantic_Segmentation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

