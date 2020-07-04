import sys
_module = sys.modules[__name__]
del sys
datasets = _module
evaluation = _module
compute_roc = _module
megaface = _module
verify = _module
main = _module
models = _module
backbones = _module
densenet = _module
inception = _module
ir = _module
lookup = _module
nasnet = _module
resnet = _module
vgg = _module
ext_layers = _module
arc_fc_layer = _module
multitask = _module
tools = _module
convert_data = _module
convert_data_para = _module
ensemble = _module
param_stat = _module
split_id_even = _module
split_id_ratio = _module
visualize = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import logging


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


import torch.optim


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import math


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


import torch.distributed as dist


import random


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, feature_dim=256,
        spatial_size=224):
        spatial_map = {(224): 7, (112): 3}
        super(DenseNet, self).__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.face_feature = nn.Linear(num_features * spatial_map[
            spatial_size] * spatial_map[spatial_size], feature_dim)
        self.drop1 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = out.view(out.size(0), -1)
        out = self.face_feature(out)
        out = self.drop1(out)
        return out


class Inception3(nn.Module):

    def __init__(self, feature_dim, aux_logits=False, transform_input=False):
        super(Inception3, self).__init__()
        self.feature_dim = feature_dim
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.layer1x1 = nn.Sequential(nn.Conv2d(2048, 256, 1, 1, 0, bias=
            False), nn.BatchNorm2d(256), nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.feature = nn.Linear(6400, feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, (0)] = x[:, (0)] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, (1)] = x[:, (1)] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, (2)] = x[:, (2)] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        x = self.drop2(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features,
            kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=
            (0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding
            =(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, avgpool, feature_dim, spatial_size=224):
        super(InceptionResNetV2, self).__init__()
        self.feature_dim = feature_dim
        self.use_avgpool = avgpool
        spatial_map = {(224): 5, (112): 2}
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        if self.use_avgpool:
            self.avgpool_1a = nn.AvgPool2d(6, count_include_pad=False)
            self.feature = nn.Linear(1536, feature_dim)
        else:
            self.layer1x1 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
            self.drop1 = nn.Dropout(0.5)
            self.feature = nn.Linear(256 * spatial_map[spatial_size] *
                spatial_map[spatial_size], feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def up_to_feature(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        if self.use_avgpool:
            x = self.avgpool_1a(x)
        else:
            x = self.layer1x1(x)
            x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        x = self.drop2(x)
        return x

    def forward(self, input):
        x = self.up_to_feature(input)
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self):
        super(CellStem0, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1,
            bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(42, eps=0.001,
            momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(42, 42, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias
            =False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias
            =False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(96, 42, 5, 2, 2, bias
            =False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(42, 42, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self):
        super(CellStem1, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(168, 84, 1, stride=1,
            bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(84, eps=0.001,
            momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=
            False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=
            False))
        self.final_path_bn = nn.BatchNorm2d(84, eps=0.001, momentum=0.1,
            affine=True)
        self.comb_iter_0_left = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(84, 84, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels,
            kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetASmall(nn.Module):

    def __init__(self, feature_dim, spatial_size=224):
        super(NASNetASmall, self).__init__()
        self.feature_dim = feature_dim
        spatial_map = {(224): 7, (112): 3}
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =96, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(96, eps=0.001, momentum=
            0.1, affine=True))
        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()
        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
            in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008,
            out_channels_left=336, in_channels_right=1008,
            out_channels_right=336)
        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.relu = nn.ReLU()
        self.pool_last = MaxPoolPad()
        self.layer1x1 = nn.Sequential(nn.Conv2d(2016, 256, 1, 1, 0, bias=
            False), nn.BatchNorm2d(256), nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.face_feature = nn.Linear(256 * spatial_map[spatial_size] *
            spatial_map[spatial_size], self.feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_1, x_cell_0)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_1)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        return x_cell_7

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = self.pool_last(x)
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.face_feature(x)
        x = self.drop2(x)
        return x


class NASNetASmall2(nn.Module):

    def __init__(self, feature_dim):
        super(NASNetASmall2, self).__init__()
        self.feature_dim = feature_dim
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =96, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(96, eps=0.001, momentum=
            0.1, affine=True))
        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()
        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
            in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008,
            out_channels_left=336, in_channels_right=1008,
            out_channels_right=336)
        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.relu = nn.ReLU()
        self.pool_last = MaxPoolPad()
        self.layer1x1 = nn.Sequential(nn.Conv2d(2016, 256, 1, 1, 0, bias=
            False), nn.BatchNorm2d(256), nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.face_feature = nn.Linear(12544, self.feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_1, x_cell_0)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_1)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        return x_cell_7

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = self.pool_last(x)
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.face_feature(x)
        x = self.drop2(x)
        return x


class NASNetALarge(nn.Module):

    def __init__(self, feature_dim):
        super(NASNetALarge, self).__init__()
        self.feature_dim = feature_dim
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =96, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(96, eps=0.001, momentum=
            0.1, affine=True))
        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()
        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
            in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_2 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_3 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_4 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_5 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008,
            out_channels_left=336, in_channels_right=1008,
            out_channels_right=336)
        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_8 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_9 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_10 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_11 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=2016,
            out_channels_left=672, in_channels_right=2016,
            out_channels_right=672)
        self.cell_12 = FirstCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2688, out_channels_right=672)
        self.cell_13 = NormalCell(in_channels_left=2688, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_14 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_15 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_16 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_17 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.relu = nn.ReLU()
        self.layer1x1 = nn.Sequential(nn.Conv2d(4032, 256, 1, 1, 0, bias=
            False), nn.BatchNorm2d(256), nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.face_feature = nn.Linear(12544, self.feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.face_feature(x)
        x = self.drop2(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
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

    def __init__(self, block, layers, feature_dim, spatial_size=224):
        fc_map = {(224): 12544, (112): 4096}
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer1x1 = nn.Sequential(nn.Conv2d(512 * block.expansion, 256,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (256), nn.ReLU(inplace=False))
        self.drop1 = nn.Dropout(0.5)
        self.feature = nn.Linear(fc_map[spatial_size], feature_dim)
        self.drop2 = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3.0 / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
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
        x = self.layer1x1(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        x = self.drop2(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, feature_dim, spatial_size=224):
        super(VGG, self).__init__()
        self.features = features
        self.feature_dim = feature_dim
        spatial_map = {(224): 7, (112): 3}
        self.face_features = nn.Sequential(nn.Linear(512 * spatial_map[
            spatial_size] * spatial_map[spatial_size], 2048), nn.ReLU(
            inplace=False), nn.Dropout(0.5), nn.Linear(2048, feature_dim),
            nn.Dropout(0.5))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.face_features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def where(cond, x_1, x_2):
    cond = cond.float()
    return cond * x_1 + (1 - cond) * x_2


class ArcFullyConnected(Module):

    def __init__(self, in_features, out_features, s, m, is_pw=True, is_hard
        =False):
        super(ArcFullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.is_pw = is_pw
        self.is_hard = is_hard
        assert s > 0
        assert 0 <= m < 0.5 * math.pi
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return 'in_features={}, out_features={}, s={}, m={}'.format(self.
            in_features, self.out_features, self.s, self.m)

    def forward(self, embed, label):
        n_weight = F.normalize(self.weight, p=2, dim=1)
        n_embed = F.normalize(embed, p=2, dim=1) * self.s
        out = F.linear(n_embed, n_weight)
        score = out.gather(1, label.view(-1, 1))
        cos_y = score / self.s
        sin_y = torch.sqrt(1 - cos_y ** 2)
        arc_score = self.s * (cos_y * math.cos(self.m) - sin_y * math.sin(
            self.m))
        if self.is_pw:
            if not self.is_hard:
                arc_score = where(score > 0, arc_score, score)
            else:
                mm = math.sin(math.pi - self.m) * self.m
                th = math.cos(math.pi - self.m)
                arc_score = where(score - th > 0, arc_score, score - self.s *
                    mm)
        one_hot = Variable(torch.FloatTensor(out.shape).fill_(0))
        out += (arc_score - score) * one_hot.scatter_(1, label.view(-1, 1), 1)
        return out


class MultiTaskWithLoss(nn.Module):

    def __init__(self, backbone, num_classes, feature_dim, spatial_size,
        arc_fc=False, feat_bn=False, s=64, m=0.5, is_pw=True, is_hard=False):
        super(MultiTaskWithLoss, self).__init__()
        self.feat_bn = feat_bn
        self.basemodel = backbones.__dict__[backbone](feature_dim=
            feature_dim, spatial_size=spatial_size)
        if feat_bn:
            self.bn1d = nn.BatchNorm1d(feature_dim, affine=False, eps=2e-05,
                momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.arc_fc = arc_fc
            if not arc_fc:
                self.fcs = nn.ModuleList([nn.Linear(feature_dim,
                    num_classes[k]) for k in range(self.num_tasks)])
            else:
                self.fcs = nn.ModuleList([ArcFullyConnected(feature_dim,
                    num_classes[k], s, m, is_pw, is_hard) for k in range(
                    self.num_tasks)])

    def forward(self, input, target=None, slice_idx=None, extract_mode=False):
        feature = self.basemodel(input)
        if self.feat_bn:
            feature = self.bn1d(feature)
        if extract_mode:
            return feature
        else:
            assert feature.size(0) == target.size(0)
            assert len(slice_idx) == self.num_tasks + 1
            assert slice_idx[-1] == feature.size(0), '{} vs {}'.format(
                slice_idx[-1], feature.size(0))
            if not self.arc_fc:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k + 1], (
                    ...)]) for k in range(self.num_tasks)]
            else:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k + 1], (
                    ...)], target[slice_idx[k]:slice_idx[k + 1]]) for k in
                    range(self.num_tasks)]
            target_slice = [target[slice_idx[k]:slice_idx[k + 1]] for k in
                range(self.num_tasks)]
            return [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_XiaohangZhan_face_recognition_framework(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ArcFullyConnected(*[], **{'in_features': 4, 'out_features': 4, 's': 4, 'm': 0}), [torch.rand([4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(AvgPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Block17(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_005(self):
        self._check(Block35(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_006(self):
        self._check(Block8(*[], **{}), [torch.rand([4, 2080, 64, 64])], {})

    def test_007(self):
        self._check(BranchSeparables(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(BranchSeparablesReduction(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(BranchSeparablesStem(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(CellStem0(*[], **{}), [torch.rand([4, 96, 64, 64])], {})

    def test_011(self):
        self._check(CellStem1(*[], **{}), [torch.rand([4, 96, 4, 4]), torch.rand([4, 168, 64, 64])], {})

    @_fails_compile()
    def test_012(self):
        self._check(InceptionResNetV2(*[], **{'avgpool': 4, 'feature_dim': 4}), [torch.rand([4, 3, 256, 256])], {})

    def test_013(self):
        self._check(MaxPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(Mixed_5b(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_015(self):
        self._check(Mixed_6a(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_016(self):
        self._check(Mixed_7a(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_017(self):
        self._check(NormalCell(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(ReductionCell0(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(ReductionCell1(*[], **{'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'dw_kernel': 4, 'dw_stride': 1, 'dw_padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(VGG(*[], **{'features': _mock_layer(), 'feature_dim': 4}), [torch.rand([25088, 25088])], {})

    def test_022(self):
        self._check(_Transition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

