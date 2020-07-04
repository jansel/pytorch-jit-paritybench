import sys
_module = sys.modules[__name__]
del sys
functions = _module
imagepreprocess = _module
main = _module
model_init = _module
src = _module
network = _module
alexnet = _module
base = _module
densenet = _module
inception = _module
mpncovresnet = _module
mpncovvgg = _module
resnet = _module
vgg = _module
BCNN = _module
CBP = _module
Custom = _module
GAvP = _module
MPNCOV = _module
representation = _module
torchviz = _module
dot = _module

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


import scipy.io as sio


import torch


import torch.nn as nn


import numpy as np


import random


import time


import warnings


from torchvision import datasets


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torch.utils.model_zoo as model_zoo


import warnings as warn


import re


import torch.nn.functional as F


from collections import OrderedDict


import math


from torch.autograd import Function


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6,
            4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 
            4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def get_basemodel(modeltype, pretrained=False):
    modeltype = globals()[modeltype]
    if pretrained == False:
        warn.warn('You will use model that randomly initialized!')
    return modeltype(pretrained=pretrained)


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """

    def __init__(self, modeltype, pretrained=False):
        super(Basemodel, self).__init__()
        basemodel = get_basemodel(modeltype, pretrained)
        self.pretrained = pretrained
        if modeltype.startswith('alexnet'):
            basemodel = self._reconstruct_alexnet(basemodel)
        if modeltype.startswith('vgg'):
            basemodel = self._reconstruct_vgg(basemodel)
        if modeltype.startswith('resnet'):
            basemodel = self._reconstruct_resnet(basemodel)
        if modeltype.startswith('inception'):
            basemodel = self._reconstruct_inception(basemodel)
        if modeltype.startswith('densenet'):
            basemodel = self._reconstruct_densenet(basemodel)
        if modeltype.startswith('mpncovresnet'):
            basemodel = self._reconstruct_mpncovresnet(basemodel)
        if modeltype.startswith('mpncovvgg'):
            basemodel = self._reconstruct_mpncov_vgg(basemodel)
        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim

    def _reconstruct_alexnet(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features[:-1]
        model.representation = basemodel.features[-1]
        if self.pretrained:
            model.classifier = basemodel.classifier[-1]
        else:
            model.classifier = basemodel.classifier
        model.representation_dim = 256
        return model

    def _reconstruct_vgg(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features[:-1]
        model.representation = basemodel.features[-1]
        if self.pretrained:
            model.classifier = basemodel.classifier[-1]
        else:
            model.classifier = basemodel.classifier
        model.representation_dim = 512
        return model

    def _reconstruct_resnet(self, basemodel):
        model = nn.Module()
        model.features = nn.Sequential(*list(basemodel.children())[:-2])
        model.representation = basemodel.avgpool
        model.classifier = basemodel.fc
        model.representation_dim = basemodel.fc.weight.size(1)
        return model

    def _reconstruct_inception(self, basemodel):
        model = nn.Module()
        model.features = nn.Sequential(basemodel.Conv2d_1a_3x3, basemodel.
            Conv2d_2a_3x3, basemodel.Conv2d_2b_3x3, nn.MaxPool2d(
            kernel_size=3, stride=2), basemodel.Conv2d_3b_1x1, basemodel.
            Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2), basemodel
            .Mixed_5b, basemodel.Mixed_5c, basemodel.Mixed_5d, basemodel.
            Mixed_6a, basemodel.Mixed_6b, basemodel.Mixed_6c, basemodel.
            Mixed_6d, basemodel.Mixed_6e, basemodel.Mixed_7a, basemodel.
            Mixed_7b, basemodel.Mixed_7c)
        model.representation = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = basemodel.fc
        model.representation_dim = basemodel.fc.weight.size(1)
        return model

    def _reconstruct_densenet(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features
        model.features.add_module('last_relu', nn.ReLU(inplace=True))
        model.representation = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = basemodel.classifier
        model.representation_dim = basemodel.classifier.weight.size(1)
        return model

    def _reconstruct_mpncovresnet(self, basemodel):
        model = nn.Module()
        if self.pretrained:
            model.features = nn.Sequential(*list(basemodel.children())[:-1])
            model.representation_dim = basemodel.layer_reduce.weight.size(0)
        else:
            model.features = nn.Sequential(*list(basemodel.children())[:-4])
            model.representation_dim = basemodel.layer_reduce.weight.size(1)
        model.representation = None
        model.classifier = basemodel.fc
        return model

    def _reconstruct_mpncov_vgg(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features
        model.representation = basemodel.representation
        model.classifier = basemodel.classifier
        model.representation_dim = model.representation.output_dim
        return model

    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
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
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
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
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False
        ):
        super(Inception3, self).__init__()
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
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
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


class MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256 * (256 + 1) / 2), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)
        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features[:-1]
        self.representation = MPNCOV(input_dim=512, dimension_reduction=256,
            iterNum=5)
        self.classifier = nn.Sequential(nn.Linear(32896, 4096), nn.ReLU(
            True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.
            Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BCNN(nn.Module):
    """Bilinear Pool
        implementation of Bilinear CNN (BCNN)
        https://arxiv.org/abs/1504.07889v5

     Args:
         thresh: small positive number for computation stability
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
     """

    def __init__(self, thresh=1e-08, is_vec=True, input_dim=2048):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1.0 / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x


class CBP(nn.Module):
    """Compact Bilinear Pooling
        implementation of Compact Bilinear Pooling (CBP)
        https://arxiv.org/pdf/1511.06062.pdf

     Args:
         thresh: small positive number for computation stability
         projDim: projected dimension
         input_dim: the #channel of input feature
     """

    def __init__(self, thresh=1e-08, projDim=8192, input_dim=512):
        super(CBP, self).__init__()
        self.thresh = thresh
        self.projDim = projDim
        self.input_dim = input_dim
        self.output_dim = projDim
        torch.manual_seed(1)
        self.h_ = [torch.randint(0, self.output_dim, (self.input_dim,),
            dtype=torch.long), torch.randint(0, self.output_dim, (self.
            input_dim,), dtype=torch.long)]
        self.weights_ = [(2 * torch.randint(0, 2, (self.input_dim,)) - 1).
            float(), (2 * torch.randint(0, 2, (self.input_dim,)) - 1).float()]
        indices1 = torch.cat((torch.arange(input_dim, dtype=torch.long).
            reshape(1, -1), self.h_[0].reshape(1, -1)), dim=0)
        indices2 = torch.cat((torch.arange(input_dim, dtype=torch.long).
            reshape(1, -1), self.h_[1].reshape(1, -1)), dim=0)
        self.sparseM = [torch.sparse.FloatTensor(indices1, self.weights_[0],
            torch.Size([self.input_dim, self.output_dim])).to_dense(),
            torch.sparse.FloatTensor(indices2, self.weights_[1], torch.Size
            ([self.input_dim, self.output_dim])).to_dense()]

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        bsn = 1
        batchSize, dim, h, w = x.data.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, dim)
        y = torch.ones(batchSize, self.output_dim, device=x.device)
        for img in range(batchSize // bsn):
            segLen = bsn * h * w
            upper = batchSize * h * w
            interLarge = torch.arange(img * segLen, min(upper, (img + 1) *
                segLen), dtype=torch.long)
            interSmall = torch.arange(img * bsn, min(upper, (img + 1) * bsn
                ), dtype=torch.long)
            batch_x = x_flat[(interLarge), :]
            sketch1 = batch_x.mm(self.sparseM[0]).unsqueeze(2)
            sketch1 = torch.fft(torch.cat((sketch1, torch.zeros(sketch1.
                size(), device=x.device)), dim=2), 1)
            sketch2 = batch_x.mm(self.sparseM[1]).unsqueeze(2)
            sketch2 = torch.fft(torch.cat((sketch2, torch.zeros(sketch2.
                size(), device=x.device)), dim=2), 1)
            Re = sketch1[:, :, (0)].mul(sketch2[:, :, (0)]) - sketch1[:, :, (1)
                ].mul(sketch2[:, :, (1)])
            Im = sketch1[:, :, (0)].mul(sketch2[:, :, (1)]) + sketch1[:, :, (1)
                ].mul(sketch2[:, :, (0)])
            tmp_y = torch.ifft(torch.cat((Re.unsqueeze(2), Im.unsqueeze(2)),
                dim=2), 1)[:, :, (0)]
            y[(interSmall), :] = tmp_y.view(torch.numel(interSmall), h, w,
                self.output_dim).sum(dim=1).sum(dim=1)
        y = self._signed_sqrt(y)
        y = self._l2norm(y)
        return y


class Custom(nn.Module):

    def __init__(self, input_dim=2048):
        super(Custom, self).__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x


class GAvP(nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self, input_dim=2048):
        super(GAvP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = input_dim

    def forward(self, x):
        x = self.avgpool(x)
        return x


class Covpool(Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = -1.0 / M / M * torch.ones(M, M, device=x.device
            ) + 1.0 / M * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):

    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim
            ).repeat(batchSize, 1, 1).type(dtype)
        normA = 1.0 / 3.0 * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False,
            device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(
            batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, (0), :, :] = A.bmm(ZY)
            Z[:, (0), :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, (i - 1), :, :].bmm(Y[:, (i - 1), :, :]))
                Y[:, (i), :, :] = Y[:, (i - 1), :, :].bmm(ZY)
                Z[:, (i), :, :] = ZY.bmm(Z[:, (i - 1), :, :])
            YZY = 0.5 * Y[:, (iterN - 2), :, :].bmm(I3 - Z[:, (iterN - 2),
                :, :].bmm(Y[:, (iterN - 2), :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1
            ).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 *
            torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim
            ).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, (iterN - 2), :, :].bmm(
                Z[:, (iterN - 2), :, :])) - Z[:, (iterN - 2), :, :].bmm(Y[:,
                (iterN - 2), :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, (iterN - 2), :, :].bmm(der_postCom).bmm(Y[:,
                (iterN - 2), :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, (i), :, :].bmm(Z[:, (i), :, :])
                ZY = Z[:, (i), :, :].bmm(Y[:, (i), :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) - Z[:, (i), :, :].bmm(dldZ).bmm
                    (Z[:, (i), :, :]) - ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) - Y[:, (i), :, :].bmm(dldY).bmm
                    (Y[:, (i), :, :]) - dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[(i), :, :] += (der_postComAux[i] - grad_aux[i] / (
                normA[i] * normA[i])) * torch.ones(dim, device=x.device).diag(
                ).type(dtype)
        return grad_input, None


class Triuvec(Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device
            ).type(dtype)
        y = x[:, (index)]
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(batchSize, dim * dim, device=x.device,
            requires_grad=False).type(dtype)
        grad_input[:, (index)] = grad_output
        grad_input = grad_input.reshape(batchSize, dim, dim)
        return grad_input


class MPNCOV(nn.Module):
    """Matrix power normalized Covariance pooling (MPNCOV)
        implementation of fast MPN-COV (i.e.,iSQRT-COV)
        https://arxiv.org/abs/1712.01034

     Args:
         iterNum: #iteration of Newton-schulz method
         is_sqrt: whether perform matrix square root or not
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """

    def __init__(self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048,
        dimension_reduction=None):
        super(MPNCOV, self).__init__()
        self.iterNum = iterNum
        self.is_sqrt = is_sqrt
        self.is_vec = is_vec
        self.dr = dimension_reduction
        if self.dr is not None:
            self.conv_dr_block = nn.Sequential(nn.Conv2d(input_dim, self.dr,
                kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.
                dr), nn.ReLU(inplace=True))
        output_dim = self.dr if self.dr else input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _cov_pool(self, x):
        return Covpool.apply(x)

    def _sqrtm(self, x):
        return Sqrtm.apply(x, self.iterNum)

    def _triuvec(self, x):
        return Triuvec.apply(x)

    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        if self.is_sqrt:
            x = self._sqrtm(x)
        if self.is_vec:
            x = self._triuvec(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jiangtaoxie_fast_MPN_COV(_paritybench_base):
    pass
    def test_000(self):
        self._check(BCNN(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BasicConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(CBP(*[], **{}), [torch.rand([4, 512, 4, 512])], {})

    def test_004(self):
        self._check(Custom(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(DenseNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_006(self):
        self._check(GAvP(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(InceptionA(*[], **{'in_channels': 4, 'pool_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(InceptionB(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(InceptionC(*[], **{'in_channels': 4, 'channels_7x7': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(InceptionD(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(InceptionE(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(MPNCOV(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(VGG(*[], **{'features': _mock_layer()}), [torch.rand([25088, 25088])], {})

    @_fails_compile()
    def test_014(self):
        self._check(_DenseBlock(*[], **{'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(_DenseLayer(*[], **{'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(_Transition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

