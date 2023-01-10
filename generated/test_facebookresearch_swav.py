import sys
_module = sys.modules[__name__]
del sys
eval_linear = _module
eval_semisup = _module
hubconf = _module
main_deepclusterv2 = _module
main_swav = _module
logger = _module
multicropdataset = _module
resnet50 = _module
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


import time


from logging import getLogger


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torchvision.models.resnet import resnet50 as _resnet50


import math


import numpy as np


import torch.distributed as dist


from scipy.sparse import csr_matrix


import torch.nn.functional as F


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch='resnet50', global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == 'resnet50':
                s = 2048
            elif arch == 'resnet50w2':
                s = 4096
            elif arch == 'resnet50w4':
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == 'resnet50'
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.av_pool(x)
        if self.bn is not None:
            x = self.bn(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    __constants__ = ['downsample']

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


class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, widen=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, normalize=False, output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2norm = normalize
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(nn.Linear(num_out_filters * block.expansion, hidden_mlp), nn.BatchNorm1d(hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim))
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
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

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.eval_mode:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in inputs]), return_counts=True)[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiPrototypes,
     lambda: ([], {'output_dim': 4, 'nmb_prototypes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_facebookresearch_swav(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

