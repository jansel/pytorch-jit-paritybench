import sys
_module = sys.modules[__name__]
del sys
config = _module
genotypes = _module
latency_lookup_table = _module
model_seg = _module
operations = _module
run_latency = _module
seg_oprs = _module
slimmable_ops = _module
architect = _module
config_search = _module
dataloader = _module
eval = _module
loss = _module
model_search = _module
model_seg = _module
operations = _module
seg_metrics = _module
seg_oprs = _module
slimmable_ops = _module
train_search = _module
tools = _module
BaseDataset = _module
datasets = _module
bdd = _module
camvid = _module
cityscapes = _module
engine = _module
evaluator = _module
logger = _module
tester = _module
seg_opr = _module
loss_opr = _module
metric = _module
utils = _module
darts_utils = _module
img_utils = _module
init_func = _module
pyt_utils = _module
visualize = _module
config_train = _module
dataloader = _module
loss = _module
model_seg = _module
operations = _module
seg_metrics = _module
seg_oprs = _module
slimmable_ops = _module
test = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import time


import logging


from torch import nn


from torch.autograd import Variable


from torch.utils import data


from torch.nn import functional as F


from random import shuffle


import torch.utils


import matplotlib


from matplotlib import pyplot as plt


import torch.utils.data as data


import torch.multiprocessing as mp


import math


import warnings


import torch.distributed as dist


import torchvision


BatchNorm2d = nn.BatchNorm2d


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/    0344c5503ee55e24f0de7f37336a6e08f10976fd/    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, width_mult_list=[1.0]):
        super(USBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        self.width_mult_list = width_mult_list
        self.bn = nn.ModuleList([nn.BatchNorm2d(i, affine=True) for i in [make_divisible(self.num_features_max * width_mult) for width_mult in width_mult_list]])
        self.ratio = 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio in self.width_mult_list
        idx = self.width_mult_list.index(self.ratio)
        y = self.bn[idx](input)
        return y


class USConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, depthwise=False, bias=True, width_mult_list=[1.0]):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]) + ' in? ' + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + ' in? ' + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y


latency_lookup_table = {}


table_file_name = 'latency_lookup_table.npy'


class BasicResidual1x(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(BasicResidual1x, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), 'c_in %d, int(self.C_in * self.ratio[0]) %d' % (c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = BasicResidual1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(BasicResidual2x, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn2 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in%d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_1x(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(BasicResidual_downup_1x, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), 'c_in %d, int(self.C_in * self.ratio[0]) %d' % (c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = BasicResidual_downup_1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2)) // 2, int(x.size(3)) // 2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class BasicResidual_downup_2x(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.0]):
        super(BasicResidual_downup_2x, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2:
            self.dilation = 1
        self.ratio = 1.0, 1.0
        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn2 = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in%d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d' % (h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2)) // 2, int(x.size(3)) // 2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.0]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        if stride == 1 and slimmable:
            self.conv1 = USConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        elif stride == 2:
            self.relu = nn.ReLU(inplace=True)
            if slimmable:
                self.conv1 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.conv2 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.bn = USBatchNorm2d(C_out, width_mult_list)
            else:
                self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.bn = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        if self.stride == 1:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])
        elif self.stride == 2:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.conv2.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d' % (h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = FactorizedReduce._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out
        elif self.slimmable:
            out = self.conv1(x)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            return x


OPS = {'skip': lambda C_in, C_out, stride, slimmable, width_mult_list: FactorizedReduce(C_in, C_out, stride, slimmable, width_mult_list), 'conv': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list), 'conv_downup': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list), 'conv_2x': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list), 'conv_2x_downup': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list)}


PRIMITIVES = ['skip', 'conv', 'conv_downup', 'conv_2x', 'conv_2x_downup']


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, op_idx, stride=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, stride, slimmable=False, width_mult_list=[1.0])

    def forward(self, x):
        return self._op(x)

    def forward_latency(self, size):
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out


class Cell(nn.Module):

    def __init__(self, op_idx, C_in, C_out, down):
        super(Cell, self).__init__()
        self._C_in = C_in
        self._C_out = C_out
        self._down = down
        if self._down:
            self._op = MixedOp(C_in, C_out, op_idx, stride=2)
        else:
            self._op = MixedOp(C_in, C_out, op_idx)

    def forward(self, input):
        out = self._op(input)
        return out

    def forward_latency(self, size):
        out = self._op.forward_latency(size)
        return out


class ConvNorm(nn.Module):
    """
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    """

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.0]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.0))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0
        if slimmable:
            self.conv = nn.Sequential(USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list), USBatchNorm2d(C_out, width_mult_list), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias), BatchNorm2d(C_out), nn.ReLU(inplace=True))

    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv[0].set_ratio(ratio)
        self.conv[1].set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), 'c_in %d, self.C_in * self.ratio[0] %d' % (c_in, self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, 'c_in %d, self.C_in %d' % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in
            w_out = w_in
        else:
            h_out = h_in // 2
            w_out = w_in // 2
        name = 'ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d' % (h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            None
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        assert x.size()[1] == self.C_in, '{} {}'.format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class ConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class FeatureFusion(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=1, Fch=16, scale=4, branch=2, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0, has_bn=False, norm_layer=norm_layer, has_relu=True, has_bias=False), ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0, has_bn=False, norm_layer=norm_layer, has_relu=False, has_bias=False), nn.Sigmoid())
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    @staticmethod
    def _latency(h, w, C_in, C_out):
        layer = FeatureFusion(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        name = 'ff_H%d_W%d_C%d' % (size[1], size[2], size[0])
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, size
        else:
            None
            latency = FeatureFusion._latency(size[1], size[2], self._scale * self._Fch * self._branch, self._scale * self._Fch * self._branch)
            latency_lookup_table[name] = latency
            np.save('latency_lookup_table.npy', latency_lookup_table)
            return latency, size

    def forward(self, fm):
        fm = self.conv_1x1(fm)
        return fm


class Head(nn.Module):

    def __init__(self, in_planes, out_planes=19, Fch=16, scale=4, branch=2, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(Head, self).__init__()
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            if is_aux:
                mid_planes = in_planes
            else:
                mid_planes = in_planes
        elif is_aux:
            mid_planes = in_planes // 2
        else:
            mid_planes = in_planes // 2
        self.conv_3x3 = ConvBnRelu(in_planes, mid_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.conv_1x1 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    @staticmethod
    def _latency(h, w, C_in, C_out=19):
        layer = Head(C_in, C_out)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        assert size[0] == self._in_planes, 'size[0] %d, self._in_planes %d' % (size[0], self._in_planes)
        name = 'head_H%d_W%d_Cin%d_Cout%d' % (size[1], size[2], size[0], self._out_planes)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
            return latency, (self._out_planes, size[1], size[2])
        else:
            None
            latency = Head._latency(size[1], size[2], self._scale * self._Fch * self._branch, self._out_planes)
            latency_lookup_table[name] = latency
            np.save('latency_lookup_table.npy', latency_lookup_table)
            return latency, (self._out_planes, size[1], size[2])

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        return output


def alphas2ops_path_width(alphas, path, widths, ignore_skip=False):
    """
    alphas: [alphas0, ..., alphas3]
    """
    assert len(path) == len(widths) + 1, 'len(path) %d, len(widths) %d' % (len(path), len(widths))
    ops = []
    path_compact = []
    widths_compact = []
    pos2alpha_skips = []
    min_len = int(np.round(len(path) / 3.0)) + path[-1] * 2
    for i in range(len(path)):
        scale = path[i]
        if ignore_skip:
            alphas[scale][i - scale][0] = -float('inf')
        op = alphas[scale][i - scale].argmax()
        if op == 0 and (i == len(path) - 1 or path[i] == path[i + 1]):
            pos2alpha_skips.append((i, F.softmax(alphas[scale][i - scale], dim=-1)[0]))
    pos_skips = [pos for pos, alpha in pos2alpha_skips]
    pos_downs = [pos for pos in range(len(path) - 1) if path[pos] < path[pos + 1]]
    if len(pos_downs) > 0:
        pos_downs.append(len(path))
        for i in range(len(pos_downs) - 1):
            pos1 = pos_downs[i]
            pos2 = pos_downs[i + 1]
            if pos1 + 1 in pos_skips and pos2 - 1 in pos_skips and pos_skips.index(pos2 - 1) - pos_skips.index(pos1 + 1) == pos2 - 1 - (pos1 + 1):
                min_skip = [1, -1]
                for j in range(pos1 + 1, pos2):
                    scale = path[j]
                    score = F.softmax(alphas[scale][j - scale], dim=-1)[0]
                    if score <= min_skip[0]:
                        min_skip = [score, j]
                alphas[path[min_skip[1]]][min_skip[1] - path[min_skip[1]]][0] = -float('inf')
    if len(pos2alpha_skips) > len(path) - min_len:
        pos2alpha_skips = sorted(pos2alpha_skips, key=lambda x: x[1], reverse=True)[:len(path) - min_len]
    pos_skips = [pos for pos, alpha in pos2alpha_skips]
    for i in range(len(path)):
        scale = path[i]
        if i < len(widths):
            width = widths[i]
        op = alphas[scale][i - scale].argmax()
        if op == 0:
            if i in pos_skips:
                if i == len(path) - 1:
                    widths_compact = widths_compact[:-1]
                continue
            else:
                alphas[scale][i - scale][0] = -float('inf')
                op = alphas[scale][i - scale].argmax()
        path_compact.append(scale)
        if i < len(widths):
            widths_compact.append(width)
        ops.append(op)
    assert len(path_compact) >= min_len
    return ops, path_compact, widths_compact


def downs2path(downs):
    path = [0]
    for down in downs[:-1]:
        if down == 0:
            path.append(path[-1])
        elif down == 1:
            path.append(path[-1] + 1)
    return path


def betas2path(betas, last, layers):
    downs = [0] * layers
    if last == 1:
        down_idx = np.argmax([beta[0] for beta in betas[1][1:-1].cpu().numpy()]) + 1
        downs[down_idx] = 1
    elif last == 2:
        max_prob = 0
        max_ij = 0, 1
        for j in range(layers - 4):
            for i in range(1, j - 1):
                prob = betas[1][i][0] * betas[2][j][0]
                if prob > max_prob:
                    max_ij = i, j
                    max_prob = prob
        downs[max_ij[0] + 1] = 1
        downs[max_ij[1] + 2] = 1
    path = downs2path(downs)
    assert path[-1] == last
    return path


def path2downs(path):
    """
    0 same 1 down
    """
    downs = []
    prev = path[0]
    for node in path[1:]:
        assert node - prev in [0, 1]
        if node > prev:
            downs.append(1)
        else:
            downs.append(0)
        prev = node
    downs.append(0)
    return downs


def path2widths(path, ratios, width_mult_list):
    widths = []
    for layer in range(1, len(path)):
        scale = path[layer]
        if scale == 0:
            widths.append(width_mult_list[ratios[scale][layer - 1].argmax()])
        else:
            widths.append(width_mult_list[ratios[scale][layer - scale].argmax()])
    return widths


def network_metas(alphas, betas, ratios, width_mult_list, layers, last, ignore_skip=False):
    betas[1] = F.softmax(betas[1], dim=-1)
    betas[2] = F.softmax(betas[2], dim=-1)
    path = betas2path(betas, last, layers)
    widths = path2widths(path, ratios, width_mult_list)
    ops, path, widths = alphas2ops_path_width(alphas, path, widths, ignore_skip=ignore_skip)
    assert len(ops) == len(path) and len(path) == len(widths) + 1, 'op %d, path %d, width%d' % (len(ops), len(path), len(widths))
    downs = path2downs(path)
    return ops, path, downs, widths


class Network_Multi_Path_Infer(nn.Module):

    def __init__(self, alphas, betas, ratios, num_classes=19, layers=9, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12, width_mult_list=[1.0], stem_head_width=(1.0, 1.0), ignore_skip=False):
        super(Network_Multi_Path_Infer, self).__init__()
        self._num_classes = num_classes
        assert layers >= 2
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        if ratios[0].size(1) == 1:
            if ignore_skip:
                self._width_mult_list = [1.0]
            else:
                self._width_mult_list = [4.0 / 12]
        else:
            self._width_mult_list = width_mult_list
        self._stem_head_width = stem_head_width
        self.latency = 0
        self.stem = nn.Sequential(ConvNorm(3, self.num_filters(2, stem_head_width[0]) * 2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False), BasicResidual2x(self.num_filters(2, stem_head_width[0]) * 2, self.num_filters(4, stem_head_width[0]) * 2, kernel_size=3, stride=2, groups=1, slimmable=False), BasicResidual2x(self.num_filters(4, stem_head_width[0]) * 2, self.num_filters(8, stem_head_width[0]), kernel_size=3, stride=2, groups=1, slimmable=False))
        self.ops0, self.path0, self.downs0, self.widths0 = network_metas(alphas, betas, ratios, self._width_mult_list, layers, 0, ignore_skip=ignore_skip)
        self.ops1, self.path1, self.downs1, self.widths1 = network_metas(alphas, betas, ratios, self._width_mult_list, layers, 1, ignore_skip=ignore_skip)
        self.ops2, self.path2, self.downs2, self.widths2 = network_metas(alphas, betas, ratios, self._width_mult_list, layers, 2, ignore_skip=ignore_skip)

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def build_structure(self, lasts):
        self._branch = len(lasts)
        self.lasts = lasts
        self.ops = [getattr(self, 'ops%d' % last) for last in lasts]
        self.paths = [getattr(self, 'path%d' % last) for last in lasts]
        self.downs = [getattr(self, 'downs%d' % last) for last in lasts]
        self.widths = [getattr(self, 'widths%d' % last) for last in lasts]
        self.branch_groups, self.cells = self.get_branch_groups_cells(self.ops, self.paths, self.downs, self.widths, self.lasts)
        self.build_arm_ffm_head()

    def build_arm_ffm_head(self):
        if self.training:
            if 2 in self.lasts:
                self.heads32 = Head(self.num_filters(32, self._stem_head_width[1]), self._num_classes, True, norm_layer=BatchNorm2d)
                if 1 in self.lasts:
                    self.heads16 = Head(self.num_filters(16, self._stem_head_width[1]) + self.ch_16, self._num_classes, True, norm_layer=BatchNorm2d)
                else:
                    self.heads16 = Head(self.ch_16, self._num_classes, True, norm_layer=BatchNorm2d)
            else:
                self.heads16 = Head(self.num_filters(16, self._stem_head_width[1]), self._num_classes, True, norm_layer=BatchNorm2d)
        self.heads8 = Head(self.num_filters(8, self._stem_head_width[1]) * self._branch, self._num_classes, Fch=self._Fch, scale=4, branch=self._branch, is_aux=False, norm_layer=BatchNorm2d)
        if 2 in self.lasts:
            self.arms32 = nn.ModuleList([ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]), 1, 1, 0, slimmable=False), ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False)])
            self.refines32 = nn.ModuleList([ConvNorm(self.num_filters(16, self._stem_head_width[1]) + self.ch_16, self.num_filters(16, self._stem_head_width[1]), 3, 1, 1, slimmable=False), ConvNorm(self.num_filters(8, self._stem_head_width[1]) + self.ch_8_2, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, slimmable=False)])
        if 1 in self.lasts:
            self.arms16 = ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False)
            self.refines16 = ConvNorm(self.num_filters(8, self._stem_head_width[1]) + self.ch_8_1, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, slimmable=False)
        self.ffm = FeatureFusion(self.num_filters(8, self._stem_head_width[1]) * self._branch, self.num_filters(8, self._stem_head_width[1]) * self._branch, reduction=1, Fch=self._Fch, scale=8, branch=self._branch, norm_layer=BatchNorm2d)

    def get_branch_groups_cells(self, ops, paths, downs, widths, lasts):
        num_branch = len(ops)
        layers = max([len(path) for path in paths])
        groups_all = []
        self.ch_16 = 0
        self.ch_8_2 = 0
        self.ch_8_1 = 0
        cells = nn.ModuleDict()
        branch_connections = np.ones((num_branch, num_branch))
        for l in range(layers):
            connections = np.ones((num_branch, num_branch))
            for i in range(num_branch):
                for j in range(i + 1, num_branch):
                    if len(paths[i]) <= l + 1 or len(paths[j]) <= l + 1 or paths[i][l + 1] != paths[j][l + 1] or ops[i][l] != ops[j][l] or widths[i][l] != widths[j][l]:
                        connections[i, j] = connections[j, i] = 0
            branch_connections *= connections
            branch_groups = []
            for branch in range(num_branch):
                if len(paths[branch]) < l + 1:
                    continue
                inserted = False
                for group in branch_groups:
                    if branch_connections[group[0], branch] == 1:
                        group.append(branch)
                        inserted = True
                        continue
                if not inserted:
                    branch_groups.append([branch])
            for group in branch_groups:
                if len(group) >= 2:
                    assert ops[group[0]][l] == ops[group[1]][l] and paths[group[0]][l + 1] == paths[group[1]][l + 1] and downs[group[0]][l] == downs[group[1]][l] and widths[group[0]][l] == widths[group[1]][l]
                if len(group) == 3:
                    assert ops[group[1]][l] == ops[group[2]][l] and paths[group[1]][l + 1] == paths[group[2]][l + 1] and downs[group[1]][l] == downs[group[2]][l] and widths[group[1]][l] == widths[group[2]][l]
                op = ops[group[0]][l]
                scale = 2 ** (paths[group[0]][l] + 3)
                down = downs[group[0]][l]
                if l < len(paths[group[0]]) - 1:
                    assert down == paths[group[0]][l + 1] - paths[group[0]][l]
                assert down in [0, 1]
                if l == 0:
                    cell = Cell(op, self.num_filters(scale, self._stem_head_width[0]), self.num_filters(scale * (down + 1), widths[group[0]][l]), down)
                elif l == len(paths[group[0]]) - 1:
                    assert down == 0
                    cell = Cell(op, self.num_filters(scale, widths[group[0]][l - 1]), self.num_filters(scale, self._stem_head_width[1]), down)
                else:
                    cell = Cell(op, self.num_filters(scale, widths[group[0]][l - 1]), self.num_filters(scale * (down + 1), widths[group[0]][l]), down)
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 16:
                    self.ch_16 = cell._C_in
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 8:
                    self.ch_8_2 = cell._C_in
                if 1 in self.lasts and self.lasts.index(1) in group and down and scale == 8:
                    self.ch_8_1 = cell._C_in
                for branch in group:
                    cells[str(l) + '-' + str(branch)] = cell
            groups_all.append(branch_groups)
        return groups_all, cells

    def agg_ffm(self, outputs8, outputs16, outputs32):
        pred32 = []
        pred16 = []
        pred8 = []
        for branch in range(self._branch):
            last = self.lasts[branch]
            if last == 2:
                if self.training:
                    pred32.append(outputs32[branch])
                out = self.arms32[0](outputs32[branch])
                out = F.interpolate(out, size=(outputs16[branch].size(2), outputs16[branch].size(3)), mode='bilinear', align_corners=True)
                out = self.refines32[0](torch.cat([out, outputs16[branch]], dim=1))
                if self.training:
                    pred16.append(outputs16[branch])
                out = self.arms32[1](out)
                out = F.interpolate(out, size=(outputs8[branch].size(2), outputs8[branch].size(3)), mode='bilinear', align_corners=True)
                out = self.refines32[1](torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 1:
                if self.training:
                    pred16.append(outputs16[branch])
                out = self.arms16(outputs16[branch])
                out = F.interpolate(out, size=(outputs8[branch].size(2), outputs8[branch].size(3)), mode='bilinear', align_corners=True)
                out = self.refines16(torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 0:
                pred8.append(outputs8[branch])
        if len(pred32) > 0:
            pred32 = self.heads32(torch.cat(pred32, dim=1))
        else:
            pred32 = None
        if len(pred16) > 0:
            pred16 = self.heads16(torch.cat(pred16, dim=1))
        else:
            pred16 = None
        pred8 = self.heads8(self.ffm(torch.cat(pred8, dim=1)))
        if self.training:
            return pred8, pred16, pred32
        else:
            return pred8

    def forward(self, input):
        _, _, H, W = input.size()
        stem = self.stem(input)
        outputs8 = [stem] * self._branch
        outputs16 = [stem] * self._branch
        outputs32 = [stem] * self._branch
        outputs = [stem] * self._branch
        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                output = self.cells[str(layer) + '-' + str(group[0])](outputs[group[0]])
                scale = int(H // output.size(2))
                for branch in group:
                    outputs[branch] = output
                    if scale == 8:
                        outputs8[branch] = output
                    elif scale == 16:
                        outputs16[branch] = output
                    elif scale == 32:
                        outputs32[branch] = output
        if self.training:
            pred8, pred16, pred32 = self.agg_ffm(outputs8, outputs16, outputs32)
            pred8 = F.interpolate(pred8, scale_factor=8, mode='bilinear', align_corners=True)
            if pred16 is not None:
                pred16 = F.interpolate(pred16, scale_factor=16, mode='bilinear', align_corners=True)
            if pred32 is not None:
                pred32 = F.interpolate(pred32, scale_factor=32, mode='bilinear', align_corners=True)
            return pred8, pred16, pred32
        else:
            pred8 = self.agg_ffm(outputs8, outputs16, outputs32)
            out = F.interpolate(pred8, size=(int(pred8.size(2)) * 8, int(pred8.size(3)) * 8), mode='bilinear', align_corners=True)
            return out

    def forward_latency(self, size):
        _, H, W = size
        latency_total = 0
        latency, size = self.stem[0].forward_latency(size)
        latency_total += latency
        latency, size = self.stem[1].forward_latency(size)
        latency_total += latency
        latency, size = self.stem[2].forward_latency(size)
        latency_total += latency
        outputs8 = [size] * self._branch
        outputs16 = [size] * self._branch
        outputs32 = [size] * self._branch
        outputs = [size] * self._branch
        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                latency, size = self.cells[str(layer) + '-' + str(group[0])].forward_latency(outputs[group[0]])
                latency_total += latency
                scale = int(H // size[1])
                for branch in group:
                    outputs[branch] = size
                    if scale == 4:
                        outputs4[branch] = size
                    elif scale == 16:
                        outputs16[branch] = size
                    elif scale == 32:
                        outputs32[branch] = size
        for branch in range(self._branch):
            last = self.lasts[branch]
            if last == 2:
                latency, size = self.arms32[0].forward_latency(outputs32[branch])
                latency_total += latency
                latency, size = self.refines32[0].forward_latency((size[0] + self.ch_16, size[1] * 2, size[2] * 2))
                latency_total += latency
                latency, size = self.arms32[1].forward_latency(size)
                latency_total += latency
                latency, size = self.refines32[1].forward_latency((size[0] + self.ch_8_2, size[1] * 2, size[2] * 2))
                latency_total += latency
                out_size = size
            elif last == 1:
                latency, size = self.arms16.forward_latency(outputs16[branch])
                latency_total += latency
                latency, size = self.refines16.forward_latency((size[0] + self.ch_8_1, size[1] * 2, size[2] * 2))
                latency_total += latency
                out_size = size
            elif last == 0:
                out_size = outputs8[branch]
        latency, size = self.ffm.forward_latency((out_size[0] * self._branch, out_size[1], out_size[2]))
        latency_total += latency
        latency, size = self.heads8.forward_latency(size)
        latency_total += latency
        return latency_total, size


class SeparableConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)
        return inputs


class SELayer(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_planes, out_planes // reduction), nn.ReLU(inplace=True), nn.Linear(out_planes // reduction, out_planes), nn.Sigmoid())
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2
        return fm


class BNRefine(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class RefineResidual(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class AttentionRefinement(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(out_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=False, has_bias=False), nn.Sigmoid())

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se
        return fm


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class FocalLoss(nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

    def __init__(self, gamma=2, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='mean')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class SoftCrossEntropyLoss2d(nn.Module):

    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index + 1)], targets[range(index, index + 1)]) / (targets.size()[2] * targets.size()[3])
        return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, n_min=0.1, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        n_min = int(self.n_min * len(loss))
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Network_Multi_Path(nn.Module):

    def __init__(self, num_classes=19, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12, width_mult_list=[1.0], prun_modes=['arch_ratio'], stem_head_width=[(1.0, 1.0)]):
        super(Network_Multi_Path, self).__init__()
        self._num_classes = num_classes
        assert layers >= 3
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None
        self._stem_head_width = stem_head_width
        self._flops = 0
        self._params = 0
        self.stem = nn.ModuleList([nn.Sequential(ConvNorm(3, self.num_filters(2, stem_ratio) * 2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False), BasicResidual2x(self.num_filters(2, stem_ratio) * 2, self.num_filters(4, stem_ratio) * 2, kernel_size=3, stride=2, groups=1, slimmable=False), BasicResidual2x(self.num_filters(4, stem_ratio) * 2, self.num_filters(8, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)) for stem_ratio, _ in self._stem_head_width])
        self.cells = nn.ModuleList()
        for l in range(layers):
            cells = nn.ModuleList()
            if l == 0:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
            elif l == 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
            elif l < layers - 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            else:
                cells.append(Cell(self.num_filters(8), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            self.cells.append(cells)
        self.refine32 = nn.ModuleList([nn.ModuleList([ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False), ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False), ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False), ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width])
        self.refine16 = nn.ModuleList([nn.ModuleList([ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False), ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width])
        self.head0 = nn.ModuleList([Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width])
        self.head1 = nn.ModuleList([Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width])
        self.head2 = nn.ModuleList([Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width])
        self.head02 = nn.ModuleList([Head(self.num_filters(8, head_ratio) * 2, num_classes, False) for _, head_ratio in self._stem_head_width])
        self.head12 = nn.ModuleList([Head(self.num_filters(8, head_ratio) * 2, num_classes, False) for _, head_ratio in self._stem_head_width])
        self._arch_names = []
        self._arch_parameters = []
        for i in range(len(self._prun_modes)):
            arch_name, arch_param = self._build_arch_parameters(i)
            self._arch_names.append(arch_name)
            self._arch_parameters.append(arch_param)
            self._reset_arch_parameters(i)
        self.arch_idx = 0

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def new(self):
        model_new = Network(self._num_classes, self._layers, self._criterion, self._Fch)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def sample_prun_ratio(self, mode='arch_ratio'):
        """
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        """
        assert mode in ['min', 'max', 'random', 'arch_ratio']
        if mode == 'arch_ratio':
            ratios = self._arch_names[self.arch_idx]['ratios']
            ratios0 = getattr(self, ratios[0])
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(gumbel_softmax(F.log_softmax(ratios0[layer], dim=-1), hard=True))
            ratios1 = getattr(self, ratios[1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(gumbel_softmax(F.log_softmax(ratios1[layer], dim=-1), hard=True))
            ratios2 = getattr(self, ratios[2])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(gumbel_softmax(F.log_softmax(ratios2[layer], dim=-1), hard=True))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == 'min':
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[0])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[0])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[0])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == 'max':
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[-1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[-1])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[-1])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == 'random':
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(np.random.choice(self._width_mult_list))
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(np.random.choice(self._width_mult_list))
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(np.random.choice(self._width_mult_list))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]

    def forward(self, input):
        stem = self.stem[self.arch_idx]
        refine16 = self.refine16[self.arch_idx]
        refine32 = self.refine32[self.arch_idx]
        head0 = self.head0[self.arch_idx]
        head1 = self.head1[self.arch_idx]
        head2 = self.head2[self.arch_idx]
        head02 = self.head02[self.arch_idx]
        head12 = self.head12[self.arch_idx]
        alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][0]), dim=-1)
        alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][1]), dim=-1)
        alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][2]), dim=-1)
        alphas = [alphas0, alphas1, alphas2]
        betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['betas'][0]), dim=-1)
        betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['betas'][1]), dim=-1)
        betas = [None, betas1, betas2]
        if self.prun_mode is not None:
            ratios = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])
        out_prev = [[stem(input), None]]
        for i, cells in enumerate(self.cells):
            out = []
            for j, cell in enumerate(cells):
                out0 = None
                out1 = None
                down0 = None
                down1 = None
                alpha = alphas[j][i - j]
                if i == 0 and j == 0:
                    ratio = self._stem_head_width[self.arch_idx][0], ratios[j][i - j], ratios[j + 1][i - j]
                elif i == self._layers - 1:
                    if j == 0:
                        ratio = ratios[j][i - j - 1], self._stem_head_width[self.arch_idx][1], None
                    else:
                        ratio = ratios[j][i - j], self._stem_head_width[self.arch_idx][1], None
                elif j == 2:
                    ratio = ratios[j][i - j], ratios[j][i - j + 1], None
                elif j == 0:
                    ratio = ratios[j][i - j - 1], ratios[j][i - j], ratios[j + 1][i - j]
                else:
                    ratio = ratios[j][i - j], ratios[j][i - j + 1], ratios[j + 1][i - j]
                if j == 0:
                    out1, down1 = cell(out_prev[0][0], alpha, ratio)
                    out.append((out1, down1))
                elif i == j:
                    out0, down0 = cell(out_prev[j - 1][1], alpha, ratio)
                    out.append((out0, down0))
                else:
                    if betas[j][i - j - 1][0] > 0:
                        out0, down0 = cell(out_prev[j - 1][1], alpha, ratio)
                    if betas[j][i - j - 1][1] > 0:
                        out1, down1 = cell(out_prev[j][0], alpha, ratio)
                    out.append((sum(w * out for w, out in zip(betas[j][i - j - 1], [out0, out1])), sum(w * down if down is not None else 0 for w, down in zip(betas[j][i - j - 1], [down0, down1]))))
            out_prev = out
        out0 = None
        out1 = None
        out2 = None
        out0 = out[0][0]
        out1 = F.interpolate(refine16[0](out[1][0]), scale_factor=2, mode='bilinear', align_corners=True)
        out1 = refine16[1](torch.cat([out1, out[0][0]], dim=1))
        out2 = F.interpolate(refine32[0](out[2][0]), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = refine32[1](torch.cat([out2, out[1][0]], dim=1))
        out2 = F.interpolate(refine32[2](out2), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = refine32[3](torch.cat([out2, out[0][0]], dim=1))
        pred0 = head0(out0)
        pred1 = head1(out1)
        pred2 = head2(out2)
        pred02 = head02(torch.cat([out0, out2], dim=1))
        pred12 = head12(torch.cat([out1, out2], dim=1))
        if not self.training:
            pred0 = F.interpolate(pred0, scale_factor=8, mode='bilinear', align_corners=True)
            pred1 = F.interpolate(pred1, scale_factor=8, mode='bilinear', align_corners=True)
            pred2 = F.interpolate(pred2, scale_factor=8, mode='bilinear', align_corners=True)
            pred02 = F.interpolate(pred02, scale_factor=8, mode='bilinear', align_corners=True)
            pred12 = F.interpolate(pred12, scale_factor=8, mode='bilinear', align_corners=True)
        return pred0, pred1, pred2, pred02, pred12

    def forward_latency(self, size, alpha=True, beta=True, ratio=True):
        stem = self.stem[self.arch_idx]
        if alpha:
            alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][0]), dim=-1)
            alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][1]), dim=-1)
            alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['alphas'][2]), dim=-1)
            alphas = [alphas0, alphas1, alphas2]
        else:
            alphas = [torch.ones_like(getattr(self, self._arch_names[self.arch_idx]['alphas'][0])) * 1.0 / len(PRIMITIVES), torch.ones_like(getattr(self, self._arch_names[self.arch_idx]['alphas'][1])) * 1.0 / len(PRIMITIVES), torch.ones_like(getattr(self, self._arch_names[self.arch_idx]['alphas'][2])) * 1.0 / len(PRIMITIVES)]
        if beta:
            betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['betas'][0]), dim=-1)
            betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]['betas'][1]), dim=-1)
            betas = [None, betas1, betas2]
        else:
            betas = [None, torch.ones_like(getattr(self, self._arch_names[self.arch_idx]['betas'][0])) * 1.0 / 2, torch.ones_like(getattr(self, self._arch_names[self.arch_idx]['betas'][1])) * 1.0 / 2]
        if ratio:
            if self.prun_mode is not None:
                ratios = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])
        else:
            ratios = self.sample_prun_ratio(mode='max')
        stem_latency = 0
        latency, size = stem[0].forward_latency(size)
        stem_latency = stem_latency + latency
        latency, size = stem[1].forward_latency(size)
        stem_latency = stem_latency + latency
        latency, size = stem[2].forward_latency(size)
        stem_latency = stem_latency + latency
        out_prev = [[size, None]]
        latency_total = [[stem_latency, 0], [0, 0], [0, 0]]
        for i, cells in enumerate(self.cells):
            out = []
            latency = []
            for j, cell in enumerate(cells):
                out0 = None
                out1 = None
                down0 = None
                down1 = None
                alpha = alphas[j][i - j]
                if i == 0 and j == 0:
                    ratio = self._stem_head_width[self.arch_idx][0], ratios[j][i - j], ratios[j + 1][i - j]
                elif i == self._layers - 1:
                    if j == 0:
                        ratio = ratios[j][i - j - 1], self._stem_head_width[self.arch_idx][1], None
                    else:
                        ratio = ratios[j][i - j], self._stem_head_width[self.arch_idx][1], None
                elif j == 2:
                    ratio = ratios[j][i - j], ratios[j][i - j + 1], None
                elif j == 0:
                    ratio = ratios[j][i - j - 1], ratios[j][i - j], ratios[j + 1][i - j]
                else:
                    ratio = ratios[j][i - j], ratios[j][i - j + 1], ratios[j + 1][i - j]
                if j == 0:
                    out1, down1 = cell.forward_latency(out_prev[0][0], alpha, ratio)
                    out.append((out1[1], down1[1] if down1 is not None else None))
                    latency.append([out1[0], down1[0] if down1 is not None else None])
                elif i == j:
                    out0, down0 = cell.forward_latency(out_prev[j - 1][1], alpha, ratio)
                    out.append((out0[1], down0[1] if down0 is not None else None))
                    latency.append([out0[0], down0[0] if down0 is not None else None])
                else:
                    if betas[j][i - j - 1][0] > 0:
                        out0, down0 = cell.forward_latency(out_prev[j - 1][1], alpha, ratio)
                    if betas[j][i - j - 1][1] > 0:
                        out1, down1 = cell.forward_latency(out_prev[j][0], alpha, ratio)
                    assert out0 is None and out1 is None or out0[1] == out1[1]
                    assert down0 is None and down1 is None or down0[1] == down1[1]
                    out.append((out0[1], down0[1] if down0 is not None else None))
                    latency.append([sum(w * out for w, out in zip(betas[j][i - j - 1], [out0[0], out1[0]])), sum(w * down if down is not None else 0 for w, down in zip(betas[j][i - j - 1], [down0[0] if down0 is not None else None, down1[0] if down1 is not None else None]))])
            out_prev = out
            for ii, lat in enumerate(latency):
                if ii == 0:
                    if lat[0] is not None:
                        latency_total[ii][0] = latency_total[ii][0] + lat[0]
                    if lat[1] is not None:
                        latency_total[ii][1] = latency_total[ii][0] + lat[1]
                elif i == ii:
                    if lat[0] is not None:
                        latency_total[ii][0] = latency_total[ii - 1][1] + lat[0]
                    if lat[1] is not None:
                        latency_total[ii][1] = latency_total[ii - 1][1] + lat[1]
                else:
                    if lat[0] is not None:
                        latency_total[ii][0] = betas[j][i - j - 1][1] * latency_total[ii][0] + betas[j][i - j - 1][0] * latency_total[ii - 1][1] + lat[0]
                    if lat[1] is not None:
                        latency_total[ii][1] = betas[j][i - j - 1][1] * latency_total[ii][0] + betas[j][i - j - 1][0] * latency_total[ii - 1][1] + lat[1]
        latency0 = latency_total[0][0]
        latency1 = latency_total[1][0]
        latency2 = latency_total[2][0]
        latency = sum([latency0, latency1, latency2])
        return latency

    def _loss(self, input, target, pretrain=False):
        loss = 0
        if pretrain is not True:
            self.prun_mode = None
            for idx in range(len(self._arch_names)):
                self.arch_idx = idx
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
        if len(self._width_mult_list) > 1:
            self.prun_mode = 'max'
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
            self.prun_mode = 'min'
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
            if pretrain == True:
                self.prun_mode = 'random'
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
                self.prun_mode = 'random'
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = 'max'
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
        return loss

    def _build_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)
        alphas = [('alpha_' + str(idx) + '_' + str(scale)) for scale in [0, 1, 2]]
        betas = [('beta_' + str(idx) + '_' + str(scale)) for scale in [1, 2]]
        setattr(self, alphas[0], nn.Parameter(Variable(0.001 * torch.ones(self._layers, num_ops), requires_grad=True)))
        setattr(self, alphas[1], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 1, num_ops), requires_grad=True)))
        setattr(self, alphas[2], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 2, num_ops), requires_grad=True)))
        setattr(self, betas[0], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 2, 2), requires_grad=True)))
        setattr(self, betas[1], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 3, 2), requires_grad=True)))
        ratios = [('ratio_' + str(idx) + '_' + str(scale)) for scale in [0, 1, 2]]
        if self._prun_modes[idx] == 'arch_ratio':
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        setattr(self, ratios[0], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 1, num_widths), requires_grad=True)))
        setattr(self, ratios[1], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 1, num_widths), requires_grad=True)))
        setattr(self, ratios[2], nn.Parameter(Variable(0.001 * torch.ones(self._layers - 2, num_widths), requires_grad=True)))
        return {'alphas': alphas, 'betas': betas, 'ratios': ratios}, [getattr(self, name) for name in alphas] + [getattr(self, name) for name in betas] + [getattr(self, name) for name in ratios]

    def _reset_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)
        if self._prun_modes[idx] == 'arch_ratio':
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        getattr(self, self._arch_names[idx]['alphas'][0]).data = Variable(0.001 * torch.ones(self._layers, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]['alphas'][1]).data = Variable(0.001 * torch.ones(self._layers - 1, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]['alphas'][2]).data = Variable(0.001 * torch.ones(self._layers - 2, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]['betas'][0]).data = Variable(0.001 * torch.ones(self._layers - 2, 2), requires_grad=True)
        getattr(self, self._arch_names[idx]['betas'][1]).data = Variable(0.001 * torch.ones(self._layers - 3, 2), requires_grad=True)
        getattr(self, self._arch_names[idx]['ratios'][0]).data = Variable(0.001 * torch.ones(self._layers - 1, num_widths), requires_grad=True)
        getattr(self, self._arch_names[idx]['ratios'][1]).data = Variable(0.001 * torch.ones(self._layers - 1, num_widths), requires_grad=True)
        getattr(self, self._arch_names[idx]['ratios'][2]).data = Variable(0.001 * torch.ones(self._layers - 2, num_widths), requires_grad=True)


class SigmoidFocalLoss(nn.Module):

    def __init__(self, ignore_label, gamma=2.0, alpha=0.25, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = target.ne(self.ignore_label).float()
        target = mask * target
        onehot = target.view(b, -1, 1)
        max_val = (-pred_sigmoid).clamp(min=0)
        pos_part = (1 - pred_sigmoid) ** self.gamma * (pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + ((-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())
        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class ProbOhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256, down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionRefinement,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Cell,
     lambda: ([], {'op_idx': 4, 'C_in': 4, 'C_out': 4, 'down': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'ksize': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureFusion,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Head,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixedOp,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'op_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConvBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SigmoidFocalLoss,
     lambda: ([], {'ignore_label': 4}),
     lambda: ([torch.rand([4, 16]), torch.rand([4, 4, 4])], {}),
     True),
    (SoftCrossEntropyLoss2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (USConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_VITA_Group_FasterSeg(_paritybench_base):
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

