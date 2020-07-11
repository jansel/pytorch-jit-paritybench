import sys
_module = sys.modules[__name__]
del sys
master = _module
coco_voc = _module
dataloader = _module
extract_cls_oid = _module
extract_det_oid = _module
model = _module
eval_utils = _module
models = _module
resnet = _module
resnet_mil = _module
resnet_utils = _module
utils = _module
vgg_mil = _module
opts = _module
test = _module
test_v2 = _module
train = _module

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


import random


import torch


from torchvision import transforms as trn


import math


import torch.nn as nn


from torch.autograd import Variable


import string


import time


import collections


from scipy.interpolate import interp1d


from matplotlib.pyplot import show


import matplotlib.pyplot as plt


import torch.optim as optim


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import tensorflow as tf


import torchvision


import torchvision.transforms as transforms


import itertools


import torch.utils.model_zoo as model_zoo


import torchvision.models as models


import functools


import torch.nn.init as init


class Criterion(nn.Module):

    def __init__(self):
        super(Criterion, self).__init__()
        self.loss0 = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        output0 = self.loss0(input, target.float())
        return output0


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class resnet_mil(nn.Module):

    def __init__(self, opt):
        super(resnet_mil, self).__init__()
        resnet = resnet.resnet101()
        resnet.load_state_dict(torch.load('/media/jxgu/d2tb/model/resnet/resnet101.pth'))
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv1', resnet.conv1)
        self.conv.add_module('bn1', resnet.bn1)
        self.conv.add_module('relu', resnet.relu)
        self.conv.add_module('maxpool', resnet.maxpool)
        self.conv.add_module('layer1', resnet.layer1)
        self.conv.add_module('layer2', resnet.layer2)
        self.conv.add_module('layer3', resnet.layer3)
        self.conv.add_module('layer4', resnet.layer4)
        self.l1 = nn.Sequential(nn.Linear(2048, 1000), nn.ReLU(True), nn.Dropout(0.5))
        self.att_size = 7
        self.pool_mil = nn.MaxPool2d(kernel_size=self.att_size, stride=0)

    def forward(self, img, att_size=14):
        x0 = self.conv(img)
        x = self.pool_mil(x0)
        x = x.squeeze(2).squeeze(2)
        x = self.l1(x)
        x1 = torch.add(torch.mul(x.view(x.size(0), 1000, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)
        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, (-1)], -1), 1))
        return out


class myResnet(nn.Module):

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)
        return fc, att


class vgg_mil(nn.Module):

    def __init__(self, opt):
        super(vgg_mil, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_1_1', torch.nn.ReLU())
        self.conv.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_1_2', torch.nn.ReLU())
        self.conv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_2_1', torch.nn.ReLU())
        self.conv.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_2_2', torch.nn.ReLU())
        self.conv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_3_1', torch.nn.ReLU())
        self.conv.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_3_2', torch.nn.ReLU())
        self.conv.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_3_3', torch.nn.ReLU())
        self.conv.add_module('maxpool_3', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_4_1', torch.nn.ReLU())
        self.conv.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_4_2', torch.nn.ReLU())
        self.conv.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_4_3', torch.nn.ReLU())
        self.conv.add_module('maxpool_4', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_5_1', torch.nn.ReLU())
        self.conv.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_5_2', torch.nn.ReLU())
        self.conv.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relu_5_3', torch.nn.ReLU())
        self.conv.add_module('maxpool_5', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('fc6_conv', nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0))
        self.conv.add_module('relu_6_1', torch.nn.ReLU())
        self.conv.add_module('fc7_conv', nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0))
        self.conv.add_module('relu_7_1', torch.nn.ReLU())
        self.conv.add_module('fc8_conv', nn.Conv2d(4096, 1000, kernel_size=1, stride=1, padding=0))
        self.conv.add_module('sigmoid_8', torch.nn.Sigmoid())
        self.pool_mil = nn.MaxPool2d(kernel_size=11, stride=0)
        self.weight_init()

    def weight_init(self):
        self.cnn_weight = 'model/vgg16_full_conv_mil.pth'
        self.conv.load_state_dict(torch.load(self.cnn_weight))
        None

    def forward(self, x):
        x0 = self.conv.forward(x.float())
        x = self.pool_mil(x0)
        x = x.squeeze(2).squeeze(2)
        x1 = torch.add(torch.mul(x0.view(x.size(0), 1000, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)
        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, (-1)], -1), 1))
        out = F.softmax(out)
        return out


class MIL_Precision_Score_Mapping(nn.Module):

    def __init__(self):
        super(MIL_Precision_Score_Mapping, self).__init__()
        self.mil = nn.MaxPool2d(kernel_size=11, stride=0)

    def forward(self, x, score, precision, mil_prob):
        out = self.mil(x)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Criterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MIL_Precision_Score_Mapping,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gujiuxiang_MIL_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

