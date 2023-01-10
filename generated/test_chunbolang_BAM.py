import sys
_module = sys.modules[__name__]
del sys
ASPP = _module
BAM = _module
PPM = _module
PSPNet = _module
resnet = _module
vgg = _module
test = _module
test_GFSS = _module
train = _module
train_base = _module
config = _module
dataset = _module
get_mulway_base_data = _module
get_weak_anns = _module
transform = _module
transform_tri = _module
util = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data


from torch import nn


from torch._C import device


from torch.nn import BatchNorm2d as BatchNorm


import numpy as np


import random


import time


import math


import torch.utils.model_zoo as model_zoo


import logging


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.optim


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


import copy


from torch.utils.data import Dataset


import numbers


import collections


import matplotlib.pyplot as plt


from matplotlib.pyplot import MultipleLocator


from matplotlib.ticker import FuncFormatter


from matplotlib.ticker import FormatStrFormatter


from matplotlib import font_manager


from matplotlib import rcParams


import pandas as pd


from scipy import ndimage


import torch.nn.init as initer


class ASPP(nn.Module):

    def __init__(self, out_channels=256):
        super(ASPP, self).__init__()
        self.layer6_0 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU())
        self.layer6_1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU())
        self.layer6_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True), nn.ReLU())
        self.layer6_3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True), nn.ReLU())
        self.layer6_4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True), nn.ReLU())
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat([global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        return out


class PPM(nn.Module):

    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin), nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class OneModel(nn.Module):

    def __init__(self, args):
        super(OneModel, self).__init__()
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = True
        self.classes = 16 if self.dataset == 'pascal' else 61
        assert self.layers in [50, 101, 152]
        if self.vgg:
            None
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            None
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            None
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = 1, 1
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = 1, 1
        self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
        fea_dim = 512 if self.vgg else 2048
        bins = 1, 2, 3, 6
        self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
        self.cls = nn.Sequential(nn.Conv2d(fea_dim * 2, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(p=0.1), nn.Conv2d(512, self.classes, kernel_size=1))

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD([{'params': model.encoder.parameters()}, {'params': model.ppm.parameters()}, {'params': model.cls.parameters()}], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def forward(self, x, y):
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        x = self.encoder(x)
        x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            main_loss = self.criterion(x, y.long())
            return x.max(1)[1], main_loss
        else:
            return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
            self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PPM,
     lambda: ([], {'in_dim': 4, 'reduction_dim': 4, 'bins': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_chunbolang_BAM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

