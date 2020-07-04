import sys
_module = sys.modules[__name__]
del sys
gate_layer = _module
logger = _module
main = _module
densenet_imagenet = _module
lenet = _module
preact_resnet = _module
resnet = _module
vgg_bn = _module
pruning_engine = _module
group_lasso_optimizer = _module
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


import torch


import torch.nn as nn


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import torch.backends.cudnn as cudnn


import time


import torch.distributed as dist


import torch.utils.data


import torch.utils.data.distributed


import torch.nn.parallel


import warnings


import numpy as np


import re


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import torch.optim


from copy import deepcopy


import itertools


class GateLayer(nn.Module):

    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))
        self.do_not_update = True

    def forward(self, input):
        return input * self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.input_features,
            self.output_features is not None)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        gate_types):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        if 'input' in gate_types:
            self.add_module('gate1): (input', GateLayer(num_input_features,
                num_input_features, [1, -1, 1, 1]))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        if 'output_bn' in gate_types:
            self.add_module('gate2): (output_bn', GateLayer(bn_size *
                growth_rate, bn_size * growth_rate, [1, -1, 1, 1]))
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        if 'output_conv' in gate_types:
            self.add_module('gate3): (output_conv', GateLayer(growth_rate,
                growth_rate, [1, -1, 1, 1]))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, gate_types):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate, gate_types)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, gate_types):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        if 'input' in gate_types:
            self.add_module('gate): (input', GateLayer(num_input_features,
                num_input_features, [1, -1, 1, 1]))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        if 'output_conv' in gate_types:
            self.add_module('gate): (output_conv', GateLayer(
                num_output_features, num_output_features, [1, -1, 1, 1]))


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
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        gate_types=['input', 'output_bn', 'output_conv', 'bottom', 'top']):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        if 'bottom' in gate_types:
            self.features.add_module('gate0): (bottom', GateLayer(
                num_init_features, num_init_features, [1, -1, 1, 1]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate, gate_types=gate_types)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2, gate_types=
                    gate_types)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        if 'top' in gate_types:
            self.features.add_module('gate5): (top', GateLayer(num_features,
                num_features, [1, -1, 1, 1]))
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
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size
            (0), -1)
        out = self.classifier(out)
        return out


class LeNet(nn.Module):

    def __init__(self, dataset='CIFAR10'):
        super(LeNet, self).__init__()
        if dataset == 'CIFAR10':
            nunits_input = 3
            nuintis_fc = 32 * 5 * 5
        elif dataset == 'MNIST':
            nunits_input = 1
            nuintis_fc = 32 * 4 * 4
        self.conv1 = nn.Conv2d(nunits_input, 16, 5)
        self.gate1 = GateLayer(16, 16, [1, -1, 1, 1])
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.gate2 = GateLayer(32, 32, [1, -1, 1, 1])
        self.fc1 = nn.Linear(nuintis_fc, 120)
        self.gate3 = GateLayer(120, 120, [1, -1])
        self.fc2 = nn.Linear(120, 84)
        self.gate4 = GateLayer(84, 84, [1, -1])
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.gate1(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.gate2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.gate3(out)
        out = F.relu(self.fc2(out))
        out = self.gate4(out)
        out = self.fc3(out)
        return out


def norm2d(planes, num_groups=32):
    if num_groups != 0:
        print('num_groups:{}'.format(num_groups))
    if num_groups > 0:
        return GroupNorm2D(planes, num_groups)
    else:
        return nn.BatchNorm2d(planes)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(PreActBlock, self).__init__()
        self.bn1 = norm2d(in_planes, group_norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.gate1 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.bn2 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.gate_out = GateLayer(planes, planes, [1, -1, 1, 1])
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.gate_shortcut = GateLayer(self.expansion * planes, self.
                expansion * planes, [1, -1, 1, 1])

    def forward(self, x):
        out = F.relu(self.bn1(x))
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
            shortcut = self.gate_shortcut(shortcut)
        else:
            shortcut = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.gate1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gate_out(out)
        out = out + shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(PreActBottleneck, self).__init__()
        self.bn1 = norm2d(in_planes, group_norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.gate1 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = norm2d(planes, group_norm)
        self.gate2 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.gate3 = GateLayer(self.expansion * planes, self.expansion *
            planes, [1, -1, 1, 1])
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.gate_shortcut = GateLayer(self.expansion * planes, self.
                expansion * planes, [1, -1, 1, 1])

    def forward(self, x):
        out = F.relu(self.bn1(x))
        input_out = out
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.gate1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.gate2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.gate3(out)
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
            shortcut = self.gate_shortcut(shortcut)
        else:
            shortcut = x
        out = out + shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, group_norm=0,
        dataset='CIFAR10'):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.dataset = dataset
        if dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=
                1, bias=False)
            num_classes = 10
        elif dataset == 'Imagenet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000
        self.gate_in = GateLayer(64, 64, [1, -1, 1, 1])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
            group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
            group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
            group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
            group_norm=group_norm)
        if dataset == 'CIFAR10':
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        elif dataset == 'Imagenet':
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, group_norm=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, group_norm=
                group_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.dataset == 'Imagenet':
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
        out = self.gate_in(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.dataset == 'CIFAR10':
            out = self.linear(out)
        elif self.dataset == 'Imagenet':
            out = self.fc(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, gate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.gate1 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.downsample = downsample
        self.stride = stride
        self.gate = gate

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.gate is not None:
            out = self.gate(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gate=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gate1 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gate = gate

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        if self.gate is not None:
            out = self.gate(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, skip_gate=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        gate = skip_gate
        self.gate = gate
        if gate:
            self.gate_skip64 = GateLayer(64 * 4, 64 * 4, [1, -1, 1, 1])
            self.gate_skip128 = GateLayer(128 * 4, 128 * 4, [1, -1, 1, 1])
            self.gate_skip256 = GateLayer(256 * 4, 256 * 4, [1, -1, 1, 1])
            self.gate_skip512 = GateLayer(512 * 4, 512 * 4, [1, -1, 1, 1])
            if block == BasicBlock:
                self.gate_skip64 = GateLayer(64, 64, [1, -1, 1, 1])
                self.gate_skip128 = GateLayer(128, 128, [1, -1, 1, 1])
                self.gate_skip256 = GateLayer(256, 256, [1, -1, 1, 1])
                self.gate_skip512 = GateLayer(512, 512, [1, -1, 1, 1])
        else:
            self.gate_skip64 = None
            self.gate_skip128 = None
            self.gate_skip256 = None
            self.gate_skip512 = None
        self.layer1 = self._make_layer(block, 64, layers[0], gate=self.
            gate_skip64)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            gate=self.gate_skip128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            gate=self.gate_skip256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            gate=self.gate_skip512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, gate=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, gate
            =gate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gate=gate))
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


class LinView(nn.Module):

    def __init__(self):
        super(LinView, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG(nn.Module):

    def __init__(self, features, cfg, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(cfg[0] * 7 * 7, cfg[1]),
            nn.BatchNorm1d(cfg[1]), nn.ReLU(True), nn.Linear(cfg[1], cfg[2]
            ), nn.BatchNorm1d(cfg[2]), nn.ReLU(True), nn.Linear(cfg[2],
            num_classes))
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
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_NVlabs_Taylor_pruning(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(LinView(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(PreActBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(PreActBottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

