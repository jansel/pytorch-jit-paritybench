import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
alexnet = _module
lenet = _module
mlp = _module
resnet = _module
vgg = _module
main = _module
alexnet = _module
lenet = _module
resnet = _module
resnet_b = _module
resnext = _module
vgg = _module
setup = _module
test_samplegrad = _module
torchsso = _module
autograd = _module
samplegrad = _module
curv = _module
cov = _module
batchnorm = _module
conv = _module
linear = _module
curvature = _module
fisher = _module
batchnorm = _module
conv = _module
linear = _module
hessian = _module
optim = _module
firstorder = _module
lr_scheduler = _module
secondorder = _module
vi = _module
utils = _module
accumulator = _module
chainer_communicators = _module
_utility = _module
base = _module
pure_nccl_communicator = _module
cholesky_cupy = _module
cupy = _module
inv_cupy = _module
logger = _module

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


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import torch.nn as nn


import math


import torch.distributed as dist


import torch.utils.model_zoo as model_zoo


from torch.optim import Optimizer


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


from collections import defaultdict


import numpy as np


from torch import Tensor


import warnings


import numpy


from torch.utils.dlpack import to_dlpack


from torch.utils.dlpack import from_dlpack


import scipy


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TensorAccumulator(object):

    def __init__(self):
        self._accumulation = None

    def check_type(self, data):
        accumulation = self._accumulation
        if isinstance(data, list):
            assert type(data[0]) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'
        else:
            assert type(data) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'
        if accumulation is not None:
            assert type(data) == type(accumulation), 'the type of data ({}) is different from the type of the accumulation ({})'.format(type(data), type(accumulation))

    def update(self, data, scale=1.0):
        self.check_type(data)
        accumulation = self._accumulation
        if isinstance(data, list):
            if accumulation is None:
                self._accumulation = [d.mul(scale) for d in data]
            else:
                self._accumulation = [acc.add(scale, d) for acc, d in zip(accumulation, data)]
        elif accumulation is None:
            self._accumulation = data.mul(scale)
        else:
            self._accumulation = accumulation.add(scale, data)

    def get(self, clear=True):
        accumulation = self._accumulation
        if accumulation is None:
            return
        if isinstance(accumulation, list):
            data = [d.clone() for d in self._accumulation]
        else:
            data = accumulation.clone()
        if clear:
            self.clear()
        return data

    def clear(self):
        self._accumulation = None


class AlexNetMCDropout(AlexNet):

    def __init__(self, num_classes=10, dropout_ratio=0.5, val_mc=10):
        super(AlexNetMCDropout, self).__init__(num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        dropout_ratio = self.dropout_ratio
        x = F.relu(F.dropout(self.conv1(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.dropout(self.conv2(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.dropout(self.conv3(x), p=dropout_ratio), inplace=True)
        x = F.relu(F.dropout(self.conv4(x), p=dropout_ratio), inplace=True)
        x = F.relu(F.dropout(self.conv5(x), p=dropout_ratio), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def mc_prediction(self, x):
        acc_prob = TensorAccumulator()
        m = self.val_mc
        for _ in range(m):
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1 / m)
        prob = acc_prob.get()
        return prob


class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet5MCDropout(LeNet5):

    def __init__(self, num_classes=10, dropout_ratio=0.1, val_mc=10):
        super(LeNet5MCDropout, self).__init__(num_classes=num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        p = self.dropout_ratio
        out = F.relu(F.dropout(self.conv1(x), p))
        out = F.max_pool2d(out, 2)
        out = F.relu(F.dropout(self.conv2(out), p))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(F.dropout(self.fc1(out), p))
        out = F.relu(F.dropout(self.fc2(out), p))
        out = F.dropout(self.fc3(out), p)
        return out

    def mc_prediction(self, x):
        acc_prob = TensorAccumulator()
        m = self.val_mc
        for _ in range(m):
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1 / m)
        prob = acc_prob.get()
        return prob


class LeNet5BatchNorm(nn.Module):

    def __init__(self, num_classes=10, affine=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6, affine=affine)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=affine)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=affine)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn3(self.fc1(out)))
        out = F.relu(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out


class MLP(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1000
        n_out = 10
        self.l1 = nn.Linear(28 * 28, n_hid)
        self.l2 = nn.Linear(n_hid, n_hid)
        self.l3 = nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor):
        x = x.view([-1, 28 * 28])
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, norm_stat_momentum=0.1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=norm_stat_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=norm_stat_momentum)
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, norm_stat_momentum=0.1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, momentum=norm_stat_momentum)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width, momentum=norm_stat_momentum)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=norm_stat_momentum)
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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None, norm_stat_momentum=0.1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, momentum=norm_stat_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, norm_stat_momentum=norm_stat_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, norm_stat_momentum=norm_stat_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, norm_stat_momentum=norm_stat_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, norm_stat_momentum=norm_stat_momentum)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, norm_stat_momentum=0.1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion, momentum=norm_stat_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, norm_layer, norm_stat_momentum))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=norm_layer, norm_stat_momentum=norm_stat_momentum))
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

    def __init__(self, num_classes=10, vgg_name='VGG19'):
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


class AlexNet2(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Block(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * group_width))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGG19(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)), inplace=True)
        h = F.relu(self.bn1_2(self.conv1_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn2_1(self.conv2_1(h)), inplace=True)
        h = F.relu(self.bn2_2(self.conv2_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn3_1(self.conv3_1(h)), inplace=True)
        h = F.relu(self.bn3_2(self.conv3_2(h)), inplace=True)
        h = F.relu(self.bn3_3(self.conv3_3(h)), inplace=True)
        h = F.relu(self.bn3_4(self.conv3_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn4_1(self.conv4_1(h)), inplace=True)
        h = F.relu(self.bn4_2(self.conv4_2(h)), inplace=True)
        h = F.relu(self.bn4_3(self.conv4_3(h)), inplace=True)
        h = F.relu(self.bn4_4(self.conv4_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn5_1(self.conv5_1(h)), inplace=True)
        h = F.relu(self.bn5_2(self.conv5_2(h)), inplace=True)
        h = F.relu(self.bn5_3(self.conv5_3(h)), inplace=True)
        h = F.relu(self.bn5_4(self.conv5_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out


class VGG19MCDropout(VGG19):

    def __init__(self, num_classes=10, dropout_ratio=0.1, val_mc=10):
        super(VGG19MCDropout, self).__init__(num_classes)
        self.dropout_ratio = dropout_ratio
        self.val_mc = val_mc

    def forward(self, x):
        p = self.dropout_ratio
        h = F.relu(self.bn1_1(F.dropout(self.conv1_1(x), p)), inplace=True)
        h = F.relu(self.bn1_2(F.dropout(self.conv1_2(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn2_1(F.dropout(self.conv2_1(h), p)), inplace=True)
        h = F.relu(self.bn2_2(F.dropout(self.conv2_2(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn3_1(F.dropout(self.conv3_1(h), p)), inplace=True)
        h = F.relu(self.bn3_2(F.dropout(self.conv3_2(h), p)), inplace=True)
        h = F.relu(self.bn3_3(F.dropout(self.conv3_3(h), p)), inplace=True)
        h = F.relu(self.bn3_4(F.dropout(self.conv3_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn4_1(F.dropout(self.conv4_1(h), p)), inplace=True)
        h = F.relu(self.bn4_2(F.dropout(self.conv4_2(h), p)), inplace=True)
        h = F.relu(self.bn4_3(F.dropout(self.conv4_3(h), p)), inplace=True)
        h = F.relu(self.bn4_4(F.dropout(self.conv4_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn5_1(F.dropout(self.conv5_1(h), p)), inplace=True)
        h = F.relu(self.bn5_2(F.dropout(self.conv5_2(h), p)), inplace=True)
        h = F.relu(self.bn5_3(F.dropout(self.conv5_3(h), p)), inplace=True)
        h = F.relu(self.bn5_4(F.dropout(self.conv5_4(h), p)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        out = F.dropout(self.fc(h), p)
        return out

    def mc_prediction(self, x):
        acc_prob = TensorAccumulator()
        m = self.val_mc
        for _ in range(m):
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1 / m)
        prob = acc_prob.get()
        return prob


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {}),
     True),
]

class Test_cybertronai_pytorch_sso(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

