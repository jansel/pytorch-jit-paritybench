import sys
_module = sys.modules[__name__]
del sys
DenseNet3D = _module
inception3d = _module

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


from collections import OrderedDict


from torch.autograd import Variable


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet3D, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(3, num_init_features, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), bias=False)), ('norm0', nn.BatchNorm3d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool3d(out, kernel_size=(1, 7, 7)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv3d(in_planes, n1x1, kernel_size=1), nn.BatchNorm3d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv3d(in_planes, n3x3red, kernel_size=1), nn.BatchNorm3d(n3x3red), nn.ReLU(True), nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1), nn.BatchNorm3d(n3x3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv3d(in_planes, n5x5red, kernel_size=1), nn.BatchNorm3d(n5x5red), nn.ReLU(True), nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1), nn.BatchNorm3d(n5x5), nn.ReLU(True), nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm3d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool3d(3, stride=1, padding=1), nn.Conv3d(in_planes, pool_planes, kernel_size=1), nn.BatchNorm3d(pool_planes), nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=400):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv3d(3, 64, kernel_size=3, stride=(2, 2, 2), padding=1), nn.BatchNorm3d(64), nn.ReLU(True), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)), nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=1), nn.BatchNorm3d(64), nn.ReLU(True), nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(192), nn.ReLU(True), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool3d(2, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
        self.convF = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool3(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool4(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = self.convF(out)
        out = out.view(out.size(0), -1)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GoogLeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 256, 256])], {}),
     True),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1': 4, 'n3x3red': 4, 'n3x3': 4, 'n5x5red': 4, 'n5x5': 4, 'pool_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_MohsenFayyaz89_T3D(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

