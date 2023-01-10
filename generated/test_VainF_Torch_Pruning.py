import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
densenet = _module
googlenet = _module
inceptionv4 = _module
mobilenetv2 = _module
nasnet = _module
preactresnet = _module
resnet = _module
resnet_tiny = _module
resnext = _module
senet = _module
swin = _module
vgg = _module
vit = _module
xception = _module
registry = _module
tools = _module
evaluator = _module
metrics = _module
utils = _module
engine = _module
evaluator = _module
metrics = _module
densenet = _module
googlenet = _module
inceptionv4 = _module
mobilenetv2 = _module
preactresnet = _module
resnet = _module
resnet_tiny = _module
resnext = _module
swin = _module
vgg = _module
vit = _module
prune = _module
prune_cifar = _module
train = _module
cifar_resnet = _module
prune_resnet18_cifar10 = _module
train_with_pruning = _module
setup = _module
test_convnext = _module
test_customized_layer = _module
test_dependency_graph = _module
test_dependency_lenet = _module
test_fully_connected_layers = _module
test_importance = _module
test_metrics = _module
test_multiple_inputs_and_outputs = _module
test_pruner = _module
test_pruning_fn = _module
test_rounding = _module
test_torchvision_models = _module
torch_pruning = _module
dependency = _module
functional = _module
structured = _module
unstructured = _module
helpers = _module
importance = _module
metric = _module
pruner = _module
basepruner = _module
batchnorm_scale_pruner = _module
magnitude_based_pruner = _module
structural_dropout_pruner = _module
structural_reg_pruner = _module
strategy = _module
utils = _module

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


import time


import math


from torch import nn


from torch import einsum


import numpy as np


from typing import Callable


from abc import ABC


from abc import abstractmethod


from typing import Union


from typing import Any


from typing import Mapping


from typing import Sequence


import numbers


import logging


import copy


import random


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


from torchvision.datasets import CIFAR10


from torchvision import transforms


from copy import deepcopy


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.cuda import amp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from torchvision.models.convnext import convnext_tiny


from torchvision.models.convnext import convnext_small


from torchvision.models.convnext import convnext_base


from torchvision.models.convnext import convnext_large


from torchvision.models import resnet18


from torchvision.models import densenet121 as entry


from torchvision.models import alexnet


from torchvision.models.resnet import resnet50


from torchvision.models.vision_transformer import vit_b_16


from torchvision.models.vision_transformer import vit_b_32


from torchvision.models.vision_transformer import vit_l_16


from torchvision.models.vision_transformer import vit_l_32


from torchvision.models.vision_transformer import vit_h_14


from torchvision.models.alexnet import alexnet


from torchvision.models.densenet import densenet121


from torchvision.models.densenet import densenet169


from torchvision.models.densenet import densenet201


from torchvision.models.densenet import densenet161


from torchvision.models.efficientnet import efficientnet_b0


from torchvision.models.efficientnet import efficientnet_b1


from torchvision.models.efficientnet import efficientnet_b2


from torchvision.models.efficientnet import efficientnet_b3


from torchvision.models.efficientnet import efficientnet_b4


from torchvision.models.efficientnet import efficientnet_b5


from torchvision.models.efficientnet import efficientnet_b6


from torchvision.models.efficientnet import efficientnet_b7


from torchvision.models.efficientnet import efficientnet_v2_s


from torchvision.models.efficientnet import efficientnet_v2_m


from torchvision.models.efficientnet import efficientnet_v2_l


from torchvision.models.googlenet import googlenet


from torchvision.models.inception import inception_v3


from torchvision.models.mnasnet import mnasnet0_5


from torchvision.models.mnasnet import mnasnet0_75


from torchvision.models.mnasnet import mnasnet1_0


from torchvision.models.mnasnet import mnasnet1_3


from torchvision.models.mobilenetv2 import mobilenet_v2


from torchvision.models.mobilenetv3 import mobilenet_v3_large


from torchvision.models.mobilenetv3 import mobilenet_v3_small


from torchvision.models.regnet import regnet_y_400mf


from torchvision.models.regnet import regnet_y_800mf


from torchvision.models.regnet import regnet_y_1_6gf


from torchvision.models.regnet import regnet_y_3_2gf


from torchvision.models.regnet import regnet_y_8gf


from torchvision.models.regnet import regnet_y_16gf


from torchvision.models.regnet import regnet_y_32gf


from torchvision.models.regnet import regnet_y_128gf


from torchvision.models.regnet import regnet_x_400mf


from torchvision.models.regnet import regnet_x_800mf


from torchvision.models.regnet import regnet_x_1_6gf


from torchvision.models.regnet import regnet_x_3_2gf


from torchvision.models.regnet import regnet_x_8gf


from torchvision.models.regnet import regnet_x_16gf


from torchvision.models.regnet import regnet_x_32gf


from torchvision.models.resnet import resnet18


from torchvision.models.resnet import resnet34


from torchvision.models.resnet import resnet101


from torchvision.models.resnet import resnet152


from torchvision.models.resnet import resnext50_32x4d


from torchvision.models.resnet import resnext101_32x8d


from torchvision.models.resnet import wide_resnet50_2


from torchvision.models.resnet import wide_resnet101_2


from torchvision.models.segmentation import fcn_resnet50


from torchvision.models.segmentation import fcn_resnet101


from torchvision.models.segmentation import deeplabv3_resnet50


from torchvision.models.segmentation import deeplabv3_resnet101


from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


from torchvision.models.segmentation import lraspp_mobilenet_v3_large


from torchvision.models.squeezenet import squeezenet1_0


from torchvision.models.squeezenet import squeezenet1_1


from torchvision.models.vgg import vgg11


from torchvision.models.vgg import vgg13


from torchvision.models.vgg import vgg16


from torchvision.models.vgg import vgg19


from torchvision.models.vgg import vgg11_bn


from torchvision.models.vgg import vgg13_bn


from torchvision.models.vgg import vgg16_bn


from torchvision.models.vgg import vgg19_bn


import typing


from enum import IntEnum


from numbers import Number


from numpy import isin


from functools import reduce


from abc import abstractstaticmethod


from typing import Tuple


from typing import Dict


import abc


from abc import abstractclassmethod


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Transition(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential()
        for index in range(len(nblocks) - 1):
            self.features.add_module('dense_block_layer_{}'.format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.features.add_module('dense_block{}'.format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


class Inception(nn.Module):

    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channels, n1x1, kernel_size=1), nn.BatchNorm2d(n1x1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1), nn.BatchNorm2d(n3x3_reduce), nn.ReLU(inplace=True), nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1), nn.BatchNorm2d(n3x3), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1), nn.BatchNorm2d(n5x5_reduce), nn.ReLU(inplace=True), nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5, n5x5), nn.ReLU(inplace=True), nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(input_channels, pool_proj, kernel_size=1), nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.prelayer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_Stem(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=3), BasicConv2d(32, 32, kernel_size=3, padding=1), BasicConv2d(32, 64, kernel_size=3, padding=1))
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_conv(x), self.branch3x3_pool(x)]
        x = torch.cat(x, 1)
        x = [self.branch7x7a(x), self.branch7x7b(x)]
        x = torch.cat(x, 1)
        x = [self.branchpoola(x), self.branchpoolb(x)]
        x = torch.cat(x, 1)
        return x


class InceptionA(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1), BasicConv2d(96, 96, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConv2d(input_channels, 96, kernel_size=1))

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branch1x1(x), self.branchpool(x)]
        return torch.cat(x, 1)


class ReductionA(nn.Module):

    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k, kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1), BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7stack = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)))
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), BasicConv2d(input_channels, 128, kernel_size=1))

    def forward(self, x):
        x = [self.branch1x1(x), self.branch7x7(x), self.branch7x7stack(x), self.branchpool(x)]
        return torch.cat(x, 1)


class ReductionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1))
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [self.branch3x3(x), self.branch7x7(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionC(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 384, kernel_size=1), BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)))
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConv2d(input_channels, 256, kernel_size=1))

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [self.branch3x3stacka(branch3x3stack_output), self.branch3x3stackb(branch3x3stack_output)]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)
        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [self.branch3x3a(branch3x3_output), self.branch3x3b(branch3x3_output)]
        branch3x3_output = torch.cat(branch3x3_output, 1)
        branch1x1_output = self.branch1x1(x)
        branchpool = self.branchpool(x)
        output = [branch1x1_output, branch3x3_output, branch3x3stack_output, branchpool]
        return torch.cat(output, 1)


class InceptionV4(nn.Module):

    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, num_classes=100):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)
        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module('{}_{}'.format(block.__name__, l), block(input_channels))
            input_channels = output_channels
        return layers


class InceptionResNetA(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=1), BasicConv2d(32, 48, kernel_size=3, padding=1), BasicConv2d(48, 64, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 32, kernel_size=1), BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)
        output = self.bn(shortcut + residual)
        output = self.relu(output)
        return output


class InceptionResNetB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 128, kernel_size=1), BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 1154, kernel_size=1)
        self.bn = nn.BatchNorm2d(1154)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch7x7(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shortcut = self.shortcut(x)
        output = self.bn(residual + shortcut)
        output = self.relu(output)
        return output


class InceptionResNetC(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192, kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 2048, kernel_size=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shorcut = self.shorcut(x)
        output = self.bn(shorcut + residual)
        output = self.relu(output)
        return output


class InceptionResNetReductionA(nn.Module):

    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k, kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1), BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionResNetReductionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branchpool = nn.MaxPool2d(3, stride=2)
        self.branch3x3a = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch3x3b = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 256, kernel_size=1), BasicConv2d(256, 288, kernel_size=3, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))

    def forward(self, x):
        x = [self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x), self.branchpool(x)]
        x = torch.cat(x, 1)
        return x


class InceptionResNetV2(nn.Module):

    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, num_classes=100):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(output_channels, 1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 2048, C, InceptionResNetC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x = self.linear(x)
        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module('{}_{}'.format(block.__name__, l), block(input_channels))
            input_channels = output_channels
        return layers


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, num_classes=100):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels, in_channels * t, 1), nn.BatchNorm2d(in_channels * t), nn.ReLU6(inplace=True), nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t), nn.BatchNorm2d(in_channels * t), nn.ReLU6(inplace=True), nn.Conv2d(in_channels * t, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 32, 1, padding=1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        self.conv1 = nn.Sequential(nn.Conv2d(320, 1280, 1), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True))
        self.conv2 = nn.Conv2d(1280, num_classes, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)


class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, groups=input_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeperableBranch(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__()
        self.block1 = nn.Sequential(nn.ReLU(), SeperableConv2d(input_channels, output_channels, kernel_size, **kwargs), nn.BatchNorm2d(output_channels))
        self.block2 = nn.Sequential(nn.ReLU(), SeperableConv2d(output_channels, output_channels, kernel_size, stride=1, padding=int(kernel_size / 2)), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Fit(nn.Module):
    """Make the cell outputs compatible
    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()
        self.p1 = nn.Sequential(nn.AvgPool2d(1, stride=2), nn.Conv2d(prev_filters, int(filters / 2), 1))
        self.p2 = nn.Sequential(nn.ConstantPad2d((0, 1, 0, 1), 0), nn.ConstantPad2d((-1, 0, -1, 0), 0), nn.AvgPool2d(1, stride=2), nn.Conv2d(prev_filters, int(filters / 2), 1))
        self.bn = nn.BatchNorm2d(filters)
        self.dim_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(prev_filters, filters, 1), nn.BatchNorm2d(filters))
        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)
        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)
        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()
        self.dem_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(x_in, output_channels, 1, bias=False), nn.BatchNorm2d(output_channels))
        self.block1_left = SeperableBranch(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.block1_right = nn.Sequential()
        self.block2_left = SeperableBranch(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.block2_right = SeperableBranch(output_channels, output_channels, kernel_size=5, padding=2, bias=False)
        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()
        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)
        self.block5_left = SeperableBranch(output_channels, output_channels, kernel_size=5, padding=2, bias=False)
        self.block5_right = SeperableBranch(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dem_reduce(x)
        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)
        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x


class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()
        self.dim_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(x_in, output_channels, 1), nn.BatchNorm2d(output_channels))
        self.layer1block1_left = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels, output_channels, 3, stride=1, padding=1)
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()
        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dim_reduce(x)
        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h)
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev)
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev)
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1) + self.layer2block2_right(layer1block2)
        return torch.cat([layer1block2, layer1block3, layer2block1, layer2block2], 1), x


class NasNetA(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stemfilter, 3, padding=1, bias=False), nn.BatchNorm2d(stemfilter))
        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters
        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 6, num_classes)

    def _make_normal(self, block, repeat, output):
        """make normal cell
        Args:
            block: cell type
            repeat: number of repeated normal cell
            output: output filters for each branch in normal cell
        Returns:
            stacked normal cells
        """
        layers = []
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * 6
        return layers

    def _make_reduction(self, block, output):
        """make normal cell
        Args:
            block: cell type
            output: output filters for each branch in reduction cell
        Returns:
            reduction cell
        """
        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4
        return reduction

    def _make_layers(self, repeat_cell_num, reduction_num):
        layers = []
        for i in range(reduction_num):
            layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))
        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreActBasic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class PreActBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 1, stride=stride), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class PreActResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.input_channels = 64
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = self._make_layers(block, num_block[0], 64, 1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)
        self.linear = nn.Linear(self.input_channels, num_classes)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []
        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion
        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out, feature


BASEWIDTH = 64


CARDINALITY = 32


DEPTH = 4


class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        C = CARDINALITY
        D = int(DEPTH * out_channels / BASEWIDTH)
        self.split_transforms = nn.Sequential(nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False), nn.BatchNorm2d(C * D), nn.ReLU(inplace=True), nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False), nn.BatchNorm2d(C * D), nn.ReLU(inplace=True), nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels * 4))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))


class ResNext(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride
        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)


class SqueezeExcitationLayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


BN_momentum = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16, new_resnet=False):
        super(SEBasicBlock, self).__init__()
        self.new_resnet = new_resnet
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes if new_resnet else planes, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_momentum)
        self.se = SqueezeExcitationLayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes, momentum=BN_momentum))
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.output = planes * self.expansion

    def _old_resnet(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

    def _new_resnet(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out

    def forward(self, x):
        if self.new_resnet:
            return self._new_resnet(x)
        else:
            return self._old_resnet(x)


class CifarNet(nn.Module):
    """
    This is specially designed for cifar10
    """

    def __init__(self, block, n_size, num_classes=10, reduction=16, new_resnet=False, dropout=0.0):
        super(CifarNet, self).__init__()
        self.inplane = 16
        self.new_resnet = new_resnet
        self.dropout_prob = dropout
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.dropout_prob > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob, inplace=True)
        self.fc = nn.Linear(self.inplane, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction, new_resnet=self.new_resnet))
            self.inplane = layers[-1].output
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
        if self.dropout_prob > 0:
            x = self.dropout_layer(x)
        x = self.fc(x)
        return x


class CyclicShift(nn.Module):

    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):

    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        q, k, v = map(lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d', h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)', h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):

    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):

    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4, shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding), SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4, shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding)]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):

    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0], downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1], downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2], downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3], downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_dim * 8), nn.Linear(hidden_dim * 8, num_classes))

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, return_features=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        x = F.relu(x)
        x = self.pool4(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(features)
        if return_features:
            return x, features
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


repeat = 100


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = image_height // patch_height * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class EntryFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3_residual = nn.Sequential(SeperableConv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), SeperableConv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(3, stride=2, padding=1))
        self.conv3_shortcut = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2), nn.BatchNorm2d(128))
        self.conv4_residual = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), SeperableConv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(3, stride=2, padding=1))
        self.conv4_shortcut = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2), nn.BatchNorm2d(256))
        self.conv5_residual = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(256, 728, 3, padding=1), nn.BatchNorm2d(728), nn.ReLU(inplace=True), SeperableConv2d(728, 728, 3, padding=1), nn.BatchNorm2d(728), nn.MaxPool2d(3, 1, padding=1))
        self.conv5_shortcut = nn.Sequential(nn.Conv2d(256, 728, 1), nn.BatchNorm2d(728))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut
        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(728, 728, 3, padding=1), nn.BatchNorm2d(728))
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(728, 728, 3, padding=1), nn.BatchNorm2d(728))
        self.conv3 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(728, 728, 3, padding=1), nn.BatchNorm2d(728))

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        shortcut = self.shortcut(x)
        return shortcut + residual


class MiddleFlow(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())
        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.ReLU(), SeperableConv2d(728, 728, 3, padding=1), nn.BatchNorm2d(728), nn.ReLU(), SeperableConv2d(728, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.MaxPool2d(3, stride=2, padding=1))
        self.shortcut = nn.Sequential(nn.Conv2d(728, 1024, 1, stride=2), nn.BatchNorm2d(1024))
        self.conv = nn.Sequential(SeperableConv2d(1024, 1536, 3, padding=1), nn.BatchNorm2d(1536), nn.ReLU(inplace=True), SeperableConv2d(1536, 2048, 3, padding=1), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)
        return output


class Xception(nn.Module):

    def __init__(self, block, num_classes=100):
        super().__init__()
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CustomizedLayer(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x * self.scale + self.bias

    def __repr__(self):
        return 'CustomizedLayer(in_dim=%d)' % self.in_dim


class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""

    def __init__(self, input_sizes, output_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_sizes[0], output_sizes[0])
        self.fc2 = nn.Linear(input_sizes[1], output_sizes[1])
        self.fc3 = nn.Linear(sum(output_sizes), 1000)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        x3 = F.relu(self.fc3(torch.cat([x1, x2], dim=1)))
        return x1, x2, x3


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class _CustomizedOp(nn.Module):

    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return 'CustomizedOp({})'.format(str(self.op_cls))


class _ConcatOp(nn.Module):

    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return '_ConcatOp({})'.format(self.offsets)


class DummyMHA(nn.Module):

    def __init__(self):
        super(DummyMHA, self).__init__()


class _SplitOp(nn.Module):

    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return '_SplitOp({})'.format(self.offsets)


class _ElementWiseOp(nn.Module):

    def __init__(self, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self._grad_fn = grad_fn

    def __repr__(self):
        return '_ElementWiseOp({})'.format(self._grad_fn)


class GConv(nn.Module):

    def __init__(self, gconv):
        super(GConv, self).__init__()
        self.groups = gconv.groups
        self.convs = nn.ModuleList()
        oc_size = gconv.out_channels // self.groups
        ic_size = gconv.in_channels // self.groups
        for g in range(self.groups):
            self.convs.append(nn.Conv2d(in_channels=oc_size, out_channels=ic_size, kernel_size=gconv.kernel_size, stride=gconv.stride, padding=gconv.padding, dilation=gconv.dilation, groups=1, bias=gconv.bias is not None, padding_mode=gconv.padding_mode))
        group_size = gconv.out_channels // self.groups
        gconv_weight = gconv.weight
        for i, conv in enumerate(self.convs):
            conv.weight.data = gconv_weight.data[oc_size * i:oc_size * (i + 1)]
            if gconv.bias is not None:
                conv.bias.data = gconv.bias.data[oc_size * i:oc_size * (i + 1)]

    def forward(self, x):
        split_sizes = [conv.in_channels for conv in self.convs]
        xs = torch.split(x, split_sizes, dim=1)
        out = torch.cat([conv(xi) for conv, xi in zip(self.convs, xs)], dim=1)
        return out


class StructrualDropout(nn.Module):

    def __init__(self, p):
        super(StructrualDropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        C = x.shape[1]
        if self.mask is None:
            self.mask = (torch.FloatTensor(C, device=x.device).uniform_() > self.p).view(1, -1, 1, 1)
        res = x * self.mask
        return res

    def reset(self, p):
        self.p = p
        self.mask = None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CustomizedLayer,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CyclicShift,
     lambda: ([], {'displacement': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EntryFlow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ExitFLow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 728, 64, 64])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fit,
     lambda: ([], {'prev_filters': 4, 'filters': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (FullyConnectedNet,
     lambda: ([], {'input_sizes': [4, 4], 'output_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (GoogleNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Inception,
     lambda: ([], {'input_channels': 4, 'n1x1': 4, 'n3x3_reduce': 4, 'n3x3': 4, 'n5x5_reduce': 4, 'n5x5': 4, 'pool_proj': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionA,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionB,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetA,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetB,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetC,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetReductionA,
     lambda: ([], {'input_channels': 4, 'k': 4, 'l': 4, 'm': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetReductionB,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetV2,
     lambda: ([], {'A': 4, 'B': 4, 'C': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (InceptionV4,
     lambda: ([], {'A': 4, 'B': 4, 'C': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Inception_Stem,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBottleNeck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MiddleFLowBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 728, 64, 64])], {}),
     True),
    (MiddleFlow,
     lambda: ([], {'block': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'downscaling_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBasic,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBottleNeck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReductionA,
     lambda: ([], {'input_channels': 4, 'k': 4, 'l': 4, 'm': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionB,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionCell,
     lambda: ([], {'x_in': 4, 'prev_in': 4, 'output_channels': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SEBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SeperableBranch,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeperableConv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExcitationLayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StructrualDropout,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Xception,
     lambda: ([], {'block': _mock_layer}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_VainF_Torch_Pruning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

