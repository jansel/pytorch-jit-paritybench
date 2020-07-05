import sys
_module = sys.modules[__name__]
del sys
pyprof = _module
fused_adam = _module
fused_layer_norm = _module
custom_function = _module
custom_module = _module
imagenet = _module
jit_script_function = _module
jit_script_method = _module
jit_trace_function = _module
jit_trace_method = _module
lenet = _module
operators = _module
simple = _module
resnet = _module
nvtx = _module
nvmarker = _module
parse = _module
__main__ = _module
db = _module
kernel = _module
nsight = _module
nvvp = _module
prof = _module
activation = _module
base = _module
blas = _module
conv = _module
convert = _module
data = _module
dropout = _module
embedding = _module
index_slice_join_mutate = _module
linear = _module
loss = _module
misc = _module
normalization = _module
optim = _module
output = _module
pointwise = _module
pooling = _module
prof = _module
randomSample = _module
recurrentCell = _module
reduction = _module
softmax = _module
tc = _module
usage = _module
utility = _module
L0_function_stack = _module
test_pyprof_func_stack = _module
test_lenet = _module
L0_nvtx = _module
test_pyprof_nvtx = _module
L0_pyprof_data = _module
test_pyprof_data = _module
check_copyright = _module
run_test = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.cuda.profiler as profiler


import torch.nn as nn


import torchvision.models as models


import torch.nn.functional as F


import torch.optim as optim


import torch.cuda.nvtx as nvtx


import numpy


import inspect as ins


import math


from collections import OrderedDict


from abc import ABC


from abc import abstractmethod


import inspect


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


class Foo(torch.jit.ScriptModule):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    @torch.jit.script_method
    def forward(self, input):
        return self.n * input + self.m


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4
    count = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.id = Bottleneck.count
        Bottleneck.count += 1

    def forward(self, x):
        identity = x
        nvtx.range_push('layer:Bottleneck_{}'.format(self.id))
        nvtx.range_push('layer:Conv1')
        out = self.conv1(x)
        nvtx.range_pop()
        nvtx.range_push('layer:BN1')
        out = self.bn1(out)
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_push('layer:Conv2')
        out = self.conv2(out)
        nvtx.range_pop()
        nvtx.range_push('layer:BN2')
        out = self.bn2(out)
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_push('layer:Conv3')
        out = self.conv3(out)
        nvtx.range_pop()
        nvtx.range_push('layer:BN3')
        out = self.bn3(out)
        nvtx.range_pop()
        if self.downsample is not None:
            nvtx.range_push('layer:Downsample')
            identity = self.downsample(x)
            nvtx.range_pop()
        nvtx.range_push('layer:Residual')
        out += identity
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_pop()
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        nvtx.range_push('layer:conv1_x')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv2_x')
        x = self.layer1(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv3_x')
        x = self.layer2(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv4_x')
        x = self.layer3(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv5_x')
        x = self.layer4(x)
        nvtx.range_pop()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        nvtx.range_push('layer:FC')
        x = self.fc(x)
        nvtx.range_pop()
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Foo,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeNet5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 32, 32])], {}),
     True),
]

class Test_NVIDIA_PyProf(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

