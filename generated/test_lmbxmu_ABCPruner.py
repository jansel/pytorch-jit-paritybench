import sys
_module = sys.modules[__name__]
del sys
bee_cifar = _module
bee_imagenet = _module
cifar10 = _module
cifar100 = _module
imagenet = _module
imagenet_dali = _module
get_flops_params = _module
densenet = _module
googlenet = _module
resnet = _module
resnet_cifar = _module
vgg = _module
vgg_cifar = _module
common = _module
options = _module

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


import torch.optim as optim


import time


import copy


import random


import numpy as np


import math


from torchvision.datasets import CIFAR10


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


from torchvision.datasets import CIFAR100


import torchvision.datasets as datasets


import torch.utils.data


import torch.nn.functional as F


from collections import OrderedDict


import logging


class DenseBasicBlock(nn.Module):

    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0):
        super(DenseBasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, inplanes, outplanes, filters, index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock, dropRate=0, num_classes=10, growthRate=12, compressionRate=2, filters=None, honey=None, indexes=None):
        super(DenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        if honey is None:
            self.honey = [10] * 36
        else:
            self.honey = honey
        """
        if ori == False:
            for i in range(36):
                self.honey[i] = 4
        """
        if filters == None:
            filters = []
            start = growthRate * 2
            index = 0
            for i in range(3):
                index -= 1
                filter = 0
                for j in range(n + 1):
                    if j != 0:
                        filter += int(growthRate * self.honey[index] / 10)
                    filters.append([start + filter])
                    index += 1
                start = (start + int(growthRate * self.honey[index - 1] / 10) * n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]
            indexes = []
            for f in filters:
                indexes.append(np.arange(f))
        self.growthRate = growthRate
        self.currentindex = 0
        self.dropRate = dropRate
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        self.trans1 = self._make_transition(Transition, filters[n + 1], filters[n], indexes[n])
        self.dense2 = self._make_denseblock(block, n, filters[n + 1:2 * n + 1], indexes[n + 1:2 * n + 1])
        self.trans2 = self._make_transition(Transition, filters[2 * n + 2], filters[2 * n + 1], indexes[2 * n + 1])
        self.dense3 = self._make_denseblock(block, n, filters[2 * n + 2:3 * n + 2], indexes[2 * n + 2:3 * n + 2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            self.growthRate = int(12 * self.honey[self.currentindex] / 10)
            self.currentindex += 1
            self.inplanes = filters[i]
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i], growthRate=self.growthRate, dropRate=self.dropRate))
        self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index):
        inplanes = self.inplanes
        outplanes = compressionRate
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, honey_rate, tmp_name):
        super(Inception, self).__init__()
        self.honey_rate = honey_rate
        self.tmp_name = tmp_name
        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes
        if self.n1x1:
            conv1x1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name
            self.branch1x1 = nn.Sequential(conv1x1, nn.BatchNorm2d(n1x1), nn.ReLU(True))
        if self.n3x3:
            conv3x3_1 = nn.Conv2d(in_planes, int(n3x3red * self.honey_rate / 10), kernel_size=1)
            conv3x3_2 = nn.Conv2d(int(n3x3red * self.honey_rate / 10), n3x3, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name
            self.branch3x3 = nn.Sequential(conv3x3_1, nn.BatchNorm2d(int(n3x3red * self.honey_rate / 10)), nn.ReLU(True), conv3x3_2, nn.BatchNorm2d(n3x3), nn.ReLU(True))
        if self.n5x5 > 0:
            conv5x5_1 = nn.Conv2d(in_planes, int(n5x5red * self.honey_rate / 10), kernel_size=1)
            conv5x5_2 = nn.Conv2d(int(n5x5red * self.honey_rate / 10), int(n5x5 * self.honey_rate / 10), kernel_size=3, padding=1)
            conv5x5_3 = nn.Conv2d(int(n5x5 * self.honey_rate / 10), n5x5, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name
            self.branch5x5 = nn.Sequential(conv5x5_1, nn.BatchNorm2d(int(n5x5red * self.honey_rate / 10)), nn.ReLU(True), conv5x5_2, nn.BatchNorm2d(int(n5x5 * self.honey_rate / 10)), nn.ReLU(True), conv5x5_3, nn.BatchNorm2d(n5x5), nn.ReLU(True))
        if self.pool_planes > 0:
            conv_pool = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name
            self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), conv_pool, nn.BatchNorm2d(pool_planes), nn.ReLU(True))

    def forward(self, x):
        out = []
        y1 = self.branch1x1(x)
        out.append(y1)
        y2 = self.branch3x3(x)
        out.append(y2)
        y3 = self.branch5x5(x)
        out.append(y3)
        y4 = self.branch_pool(x)
        out.append(y4)
        return torch.cat(out, 1)


cov_cfg = [(22 * i + 1) for i in range(1 + 2 + 5 + 2)]


class GoogLeNet(nn.Module):

    def __init__(self, block=Inception, filters=None, honey=None):
        super(GoogLeNet, self).__init__()
        self.covcfg = cov_cfg
        if honey is None:
            self.honey = [10] * 9
        else:
            self.honey = honey
        conv_pre = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        conv_pre.tmp_name = 'pre_layer'
        self.pre_layers = nn.Sequential(conv_pre, nn.BatchNorm2d(192), nn.ReLU(True))
        if filters is None:
            filters = [[64, 128, 32, 32], [128, 192, 96, 64], [192, 208, 48, 64], [160, 224, 64, 64], [128, 256, 64, 64], [112, 288, 64, 64], [256, 320, 128, 128], [256, 320, 128, 128], [384, 384, 128, 128]]
        self.filters = filters
        self.inception_a3 = block(192, filters[0][0], 96, filters[0][1], 16, filters[0][2], filters[0][3], self.honey[0], 'a3')
        self.inception_b3 = block(sum(filters[0]), filters[1][0], 128, filters[1][1], 32, filters[1][2], filters[1][3], self.honey[1], 'a4')
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception_a4 = block(sum(filters[1]), filters[2][0], 96, filters[2][1], 16, filters[2][2], filters[2][3], self.honey[2], 'a4')
        self.inception_b4 = block(sum(filters[2]), filters[3][0], 112, filters[3][1], 24, filters[3][2], filters[3][3], self.honey[3], 'b4')
        self.inception_c4 = block(sum(filters[3]), filters[4][0], 128, filters[4][1], 24, filters[4][2], filters[4][3], self.honey[4], 'c4')
        self.inception_d4 = block(sum(filters[4]), filters[5][0], 144, filters[5][1], 32, filters[5][2], filters[5][3], self.honey[5], 'd4')
        self.inception_e4 = block(sum(filters[5]), filters[6][0], 160, filters[6][1], 32, filters[6][2], filters[6][3], self.honey[6], 'e4')
        self.inception_a5 = block(sum(filters[6]), filters[7][0], 160, filters[7][1], 32, filters[7][2], filters[7][3], self.honey[7], 'a5')
        self.inception_b5 = block(sum(filters[7]), filters[8][0], 192, filters[8][1], 48, filters[8][2], filters[8][3], self.honey[8], 'b5')
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(sum(filters[-1]), 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.inception_a3(out)
        out = self.inception_b3(out)
        out = self.maxpool1(out)
        out = self.inception_a4(out)
        out = self.inception_b4(out)
        out = self.inception_c4(out)
        out = self.inception_d4(out)
        out = self.inception_e4(out)
        out = self.maxpool2(out)
        out = self.inception_a5(out)
        out = self.inception_b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, int(planes * honey[index] / 10), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * honey[index] / 10))
        self.conv2 = nn.Conv2d(int(planes * honey[index] / 10), planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(Bottleneck, self).__init__()
        pr_channels = int(planes * honey[index] / 10)
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)
        self.conv2 = nn.Conv2d(pr_channels, pr_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels)
        self.conv3 = nn.Conv2d(pr_channels, self.expansion * planes, kernel_size=1, bias=False)
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

    def __init__(self, block, num_layers, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.honey = honey
        self.current_conv = 0
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, honey=self.honey, index=self.current_conv, stride=stride))
        self.current_conv += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, honey=self.honey, index=self.current_conv))
            self.current_conv += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, honey, index, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        middle_planes = int(planes * honey[index] / 10)
        self.conv1 = conv3x3(inplanes, middle_planes, stride)
        self.bn1 = nn.BatchNorm2d(middle_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class BeeVGG(nn.Module):

    def __init__(self, vgg_name, honeysource):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512 * honeysource[len(honeysource) - 1] / 10), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.honeysource[x_index - Mlayers] / 10)
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseBasicBlock,
     lambda: ([], {'inplanes': 4, 'filters': 4, 'index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1': 4, 'n3x3red': 4, 'n3x3': 4, 'n5x5red': 4, 'n5x5': 4, 'pool_planes': 4, 'honey_rate': 4, 'tmp_name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'inplanes': 4, 'outplanes': 4, 'filters': 4, 'index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lmbxmu_ABCPruner(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

