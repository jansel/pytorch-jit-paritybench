import sys
_module = sys.modules[__name__]
del sys
architect = _module
architect_cost = _module
genotypes = _module
model = _module
model_search = _module
model_search_cost = _module
operations = _module
plot_loss_entropy_cost = _module
test = _module
test_imagenet = _module
train = _module
train_imagenet = _module
train_search = _module
train_search_cost = _module
train_search_modify = _module
utils = _module
visualize = _module
dist_util_torch = _module
model_edge_all = _module
model_search = _module
operations = _module
plot_5edge_cost = _module
train_search = _module
train_search_cost_entropy_loss = _module
utils = _module
blocks = _module
devkit = _module
core = _module
criterions = _module
dist_utils = _module
lr_scheduler = _module
misc = _module
utils = _module
dataset = _module
imagenet_dataset = _module
ops = _module
switchable_norm = _module
syncbn_layer = _module
syncsn_layer = _module
eval_imagenet = _module
network = _module
network_child = _module
network_eval = _module
train_imagenet = _module
utils = _module
dist_util_torch = _module
model = _module
model_edge_all = _module
model_search = _module
operations = _module
test_edge_all = _module
train = _module
train_edge_all = _module
train_imagenet = _module
train_search = _module
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


import numpy as np


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import logging


import torch.utils


import torchvision.datasets as dset


import torch.backends.cudnn as cudnn


import random


import torchvision.transforms as transforms


import time


import math


import torch.distributed as dist


from torch.nn import Module


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import Sampler


from torch.distributed import get_world_size


from torch.distributed import get_rank


from torch.utils import checkpoint as cp


from scipy import linalg


from scipy.io import loadmat


from torch.utils.data.sampler import SubsetRandomSampler


import torch.multiprocessing as mp


from torch.utils.checkpoint import checkpoint


from torch.utils.checkpoint import checkpoint_sequential


from torch.utils.data import Dataset


from torch.autograd import Function


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch.utils.data import DataLoader


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.size = 2 * C_in * (C_out // 2)
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        if self.flops == 0:
            original_h = x.size()[2]
            y = original_h // 2
            self.flops = self.size * y ** 2
            self.mac = 2 * y ** 2 * (4 * self.C_in + self.C_out // 2) + self.size
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.mac == 0:
            y = x.size()[2]
            c = x.size()[1]
            self.mac = 2 * c * y ** 2
        return x


class AvgPool(nn.Module):

    def __init__(self, C, stride, padding, count_include_pad=False, affine=True):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.C = C
        self.op = nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=count_include_pad), nn.BatchNorm2d(C, affine=affine))
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = 2 * y ** 2 * self.C
            else:
                y = original_h // 2
                self.mac = 5 * y ** 2 * self.C
            self.flops = 3 * 3 * y ** 2 * self.C
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))
        self.size = C_in * kernel_size * kernel_size + C_out * C_in
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (3 * self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (6 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


class HardZero(nn.Module):

    def __init__(self, stride):
        super(HardZero, self).__init__()
        self.stride = stride
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        return 0


class MaxPool(nn.Module):

    def __init__(self, C, stride, padding, affine=True):
        super(MaxPool, self).__init__()
        self.C = C
        self.stride = stride
        self.op = nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=padding), nn.BatchNorm2d(C, affine=affine))
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = 2 * y ** 2 * self.C
            else:
                y = original_h // 2
                self.mac = 5 * y ** 2 * self.C
            self.flops = 3 * 3 * y ** 2 * self.C
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))
        self.size = 2 * C_in * kernel_size ** 2 + (C_in + C_out) * C_in
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (7 * self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (10 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


OPS = {'none': lambda C, stride, affine: Zero(stride), 'hard_none': lambda C, stride, affine: HardZero(stride), 'avg_pool_3x3': lambda C, stride, affine: AvgPool(C, stride, 1, affine=affine), 'max_pool_3x3': lambda C, stride, affine: MaxPool(C, stride, 1, affine=affine), 'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine), 'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C, affine=affine))}


class MixedOp(nn.Module):

    def __init__(self, C, stride, op_size, op_flops, op_mac, primitives, bn_affine):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._resource_size = op_size
        self._resource_flops = op_flops
        self._resource_mac = op_mac
        self.got_flops_mac = False
        self.Primitives = primitives
        for primitive in self.Primitives:
            op = OPS[primitive](C, stride, bn_affine)
            self._resource_size[self.Primitives.index(primitive)] = op.size
            self._ops.append(op)

    def forward(self, x, weights):
        if self.got_flops_mac:
            result = sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            result = 0
            index = 0
            for w, op in zip(weights, self._ops):
                result += w * op(x)
                self._resource_flops[index] = op.flops
                self._resource_mac[index] = op.mac
                index += 1
            self.got_flops_mac = True
        return result


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))
        self.size = C_in * C_out * kernel_size * kernel_size
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (4 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives, bn_affine, use_ckpt=True):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.use_ckpt = use_ckpt
        self.Primitives = primitives
        self.bn_affine = bn_affine
        self.device = torch.device('cuda')
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=self.bn_affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=self.bn_affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=self.bn_affine)
        self._steps = steps
        self._multiplier = multiplier
        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self._num_ops = len(self.Primitives)
        self.op_size = torch.zeros(self._k, self._num_ops)
        self.op_flops = torch.zeros(self._k, self._num_ops)
        self.op_mac = torch.zeros(self._k, self._num_ops)
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.op_size[count], self.op_flops[count], self.op_mac[count], self.Primitives, self.bn_affine)
                self._ops.append(op)
                count += 1

    def forward(self, s0, s1, weights, drop_path_prob=0):
        if self.use_ckpt:
            s0 = cp.checkpoint(self.preprocess0, s0)
            s1 = cp.checkpoint(self.preprocess1, s1)
        else:
            s0 = self.preprocess0(s0)
            s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                op = self._ops[offset + j]
                if self.use_ckpt:
                    h = cp.checkpoint(op, *[h, weights[offset + j]])
                else:
                    h = op(h, weights[offset + j])
                if self.training and drop_path_prob > 0.0:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_path_prob)
                s += h
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1), self.op_size, self.op_flops, self.op_mac


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in xrange(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class MixedOpChild(nn.Module):

    def __init__(self, C, stride, name, op_size, op_flops, op_mac, primitives, bn_affine=False):
        super(MixedOpChild, self).__init__()
        self._ops = nn.ModuleList()
        self._resource_size = op_size
        self._resource_flops = op_flops
        self._resource_mac = op_mac
        self.got_flops_mac = False
        self.Primitives = primitives
        self.pos = self.Primitives.index(name)
        if 'none' in name:
            op = OPS[name](C, stride, affine=bn_affine)
            self._ops.append(op)
        else:
            for i in range(0, self.pos):
                op = OPS['hard_none'](C, stride, affine=bn_affine)
                self._ops.append(op)
            op = OPS[name](C, stride, affine=bn_affine)
            self._ops.append(op)
            if self.pos < self._resource_size.size()[0]:
                for i in range(self.pos + 1, self._resource_size.size()[0]):
                    op = OPS['hard_none'](C, stride, affine=bn_affine)
                    self._ops.append(op)

    def forward(self, x, weights):
        op = self._ops[self.pos]
        result = op(x) * weights[self.pos]
        return result


class CellChild(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives, use_ckpt, bn_affine=False):
        super(CellChild, self).__init__()
        self._steps = 4
        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.Primitives = primitives
        self._num_ops = len(self.Primitives)
        self.op_size = torch.zeros(self._k, self._num_ops)
        self.op_flops = torch.zeros(self._k, self._num_ops)
        self.op_mac = torch.zeros(self._k, self._num_ops)
        self._use_ckpt = use_ckpt
        self._reduction = reduction
        self._bn_affine = bn_affine
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=bn_affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=bn_affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=bn_affine)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = int((-3 + math.sqrt(9 + 8 * len(indices))) // 2)
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        count = 0
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = MixedOpChild(C, stride, name, self.op_size[count], self.op_flops[count], self.op_mac[count], self.Primitives, self._bn_affine)
            self._ops += [op]
            count += 1
        self._indices = indices

    def forward(self, s0, s1, drop_path_prob, weights):
        if self._use_ckpt:
            s0 = cp.checkpoint(self.preprocess0, s0)
            s1 = cp.checkpoint(self.preprocess1, s1)
        else:
            s0 = self.preprocess0(s0)
            s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                op = self._ops[offset + j]
                if self._use_ckpt:
                    h = cp.checkpoint(op, *[h, weights[offset + j]])
                else:
                    h = op(h, weights[offset + j])
                if self.training and drop_path_prob > 0.0:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_path_prob)
                s += h
            offset += len(states)
            states.append(s)
        return torch.cat(states[-4:], dim=1)


class NetworkChild(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, primitives, drop_path_prob, use_ckpt, bn_affine=False):
        super(NetworkChild, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        self.Primitives = primitives
        self._use_ckpt = use_ckpt
        self._steps = 4
        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell_use_ckpt = self._use_ckpt
            cell = CellChild(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.Primitives, cell_use_ckpt, bn_affine=bn_affine)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, normal_weights, reduce_weights):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell._reduction:
                weights = reduce_weights
            else:
                weights = normal_weights
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob, weights)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, args, rank, world_size, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.device = torch.device('cuda')
        self.snas = args.snas
        self.dsnas = args.dsnas
        self._world_size = world_size
        self._use_ckpt = args.use_ckpt
        self._resample_layer = args.resample_layer
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._resource_efficient = args.resource_efficient
        self._resource_lambda = args.resource_lambda
        self._method = args.method
        self._drop_path_prob = args.drop_path_prob
        self._normalization = args.normalization
        self._running_mean_var = args.running_mean_var
        self._separation = args.separation
        self._log_penalty = args.log_penalty
        self._loss = args.loss
        if args.ckpt_false_list != 'all':
            self._ckpt_false_list = literal_eval(args.ckpt_false_list)
        else:
            self._ckpt_false_list = range(self._layers)
        self._bn_affine = args.bn_affine
        self._distributed = args.distributed
        self._minus_baseline = args.minus_baseline
        self._loc_mean = args.loc_mean
        self._loc_std = args.loc_std
        self._temp = args.temp
        self._nsample = args.nsample
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._init_channels = args.init_channels
        self._auxiliary = args.auxiliary
        self.args = args
        self._k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.Primitives = PRIMITIVES
        self._num_ops = len(self.Primitives)
        if self._distributed:
            self.normal_log_alpha = torch.nn.Parameter(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).requires_grad_())
            self.reduce_log_alpha = torch.nn.Parameter(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std).requires_grad_())
        else:
            self.normal_log_alpha = Variable(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std), requires_grad=True)
            self.reduce_log_alpha = Variable(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std), requires_grad=True)
            self.normal_log_alpha_ema = Variable(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std), requires_grad=True)
            self.reduce_log_alpha_ema = Variable(torch.zeros(self._k, self._num_ops).normal_(self._loc_mean, self._loc_std), requires_grad=True)
        self.normal_edge_reward = torch.zeros(self.normal_log_alpha.size(0))
        self.reduce_edge_reward = torch.zeros(self.reduce_log_alpha.size(0))
        self.normal_log_alpha_pre = self.normal_log_alpha.clone().detach()
        self.reduce_log_alpha_pre = self.reduce_log_alpha.clone().detach()
        self.normal_edge_reward_running_mean = torch.zeros(1)
        self.normal_edge_reward_running_var = torch.zeros(1)
        self.reduce_edge_reward_running_mean = torch.zeros(1)
        self.reduce_edge_reward_running_var = torch.zeros(1)
        self._arch_parameters = [self.normal_log_alpha, self.reduce_log_alpha]
        self._rank = rank
        self._logger = None
        self._logging = None
        self.net_init()

    def net_init(self):
        C = self._C
        layers = self._layers
        steps = self._steps
        multiplier = self._multiplier
        stem_multiplier = self._stem_multiplier
        num_classes = self._num_classes
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.reduce_list = [layers // 3, 2 * layers // 3]
        self.num_reduce = len(self.reduce_list)
        self.num_normal = layers - self.num_reduce
        for i in range(layers):
            if i in self.reduce_list:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if self._use_ckpt:
                if i in self._ckpt_false_list:
                    cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.Primitives, self._bn_affine, use_ckpt=False)
                else:
                    cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.Primitives, self._bn_affine, use_ckpt=True)
            else:
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.Primitives, self._bn_affine, use_ckpt=False)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def logp(self, log_alpha, weights):
        lam = self._num_ops
        temp = self._temp
        epsilon = 0.0001
        epsilon_weight = torch.ones_like(weights) * epsilon
        weights_temp = torch.max(weights, epsilon_weight)
        last_term_epsilon = torch.max(weights_temp ** -temp * torch.exp(log_alpha), epsilon_weight)
        log_prob = math.log(5040) + (lam - 1) * math.log(temp) + log_alpha.sum(-1) - (temp + 1) * torch.log(weights_temp).sum(-1) - lam * torch.log(last_term_epsilon.sum(-1))
        return log_prob

    def forward(self, input, target=None, criterion=None, input_search=None, target_search=None):
        total_penalty = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        op_normal = [0, 0, 0]
        op_reduce = [0, 0, 0]
        logits_aux = None
        if not self._resample_layer:
            normal_weights = self._get_weights(self.normal_log_alpha)
            reduce_weights = self._get_weights(self.reduce_log_alpha)
            self.normal_weights = normal_weights
            self.reduce_weights = reduce_weights
        if self.args.dsnas:
            if self.args.gen_max_child_flag and not self.training:
                normal_weights = torch.zeros_like(self.normal_log_alpha).scatter_(1, torch.argmax(self.normal_log_alpha, dim=-1).view(-1, 1), 1)
                reduce_weights = torch.zeros_like(self.reduce_log_alpha).scatter_(1, torch.argmax(self.reduce_log_alpha, dim=-1).view(-1, 1), 1)
            normal_one_hot_prob = (normal_weights * F.softmax(self.normal_log_alpha, dim=-1)).sum(-1)
            reduce_one_hot_prob = (reduce_weights * F.softmax(self.reduce_log_alpha, dim=-1)).sum(-1)
            genotype_child = self.genotype_child(normal_weights, reduce_weights)
            model_child = NetworkChild(self._init_channels, self._num_classes, self._layers, self._auxiliary, genotype_child, self.Primitives, self._drop_path_prob, self._use_ckpt, self._bn_affine)
            model_child = model_child
            if not self.training:
                model_child.eval()
            self.load_child_state_dict(model_child)
            normal_weights.requires_grad_()
            reduce_weights.requires_grad_()
            logits, logits_aux = model_child(input, normal_weights, reduce_weights)
            error_loss = criterion(logits, target)
            if self.args.prox_policy_opt:
                normal_one_hot_prob_pre = (normal_weights.detach() * F.softmax(self.normal_log_alpha_pre, dim=-1)).sum(-1)
                reduce_one_hot_prob_pre = (reduce_weights.detach() * F.softmax(self.reduce_log_alpha_pre, dim=-1)).sum(-1)
                normal_ratio = normal_one_hot_prob / normal_one_hot_prob_pre
                reduce_ratio = reduce_one_hot_prob / reduce_one_hot_prob_pre
                normal_ratio_clip = torch.clamp(normal_ratio, min=1 - self.args.prox_policy_epi, max=1 + self.args.prox_policy_epi)
                reduce_ratio_clip = torch.clamp(reduce_ratio, min=1 - self.args.prox_policy_epi, max=1 + self.args.prox_policy_epi)
                loss_alpha = (torch.log(torch.max(normal_ratio, normal_ratio_clip)) + torch.log(torch.max(reduce_ratio, reduce_ratio_clip))).sum()
            else:
                loss_alpha = (torch.log(normal_one_hot_prob) + torch.log(reduce_one_hot_prob)).sum()
            if self.args.auxiliary and self.training:
                loss_aux = criterion(logits_aux, target)
                error_loss += self.args.auxiliary_weight * loss_aux
            if self.training:
                if self.args.add_entropy_loss:
                    entropy_loss = self._arch_entropy(self.normal_log_alpha) + self._arch_entropy(self.reduce_log_alpha)
                    loss = error_loss.clone() + loss_alpha.clone() + entropy_loss
                else:
                    loss = error_loss.clone() + loss_alpha.clone()
                if self.args.distributed:
                    loss.div_(self._world_size)
                for v in model_child.parameters():
                    if v.grad is not None:
                        v.grad = None
                loss.backward()
                if self.args.edge_reward_norm:
                    normal_edge_reward_mean = normal_weights.grad.sum(-1).mean()
                    normal_edge_reward_var = normal_weights.grad.sum(-1).std()
                    reduce_edge_reward_mean = reduce_weights.grad.sum(-1).mean()
                    reduce_edge_reward_var = reduce_weights.grad.sum(-1).std()
                    self.normal_edge_reward_running_mean = self.normal_edge_reward_running_mean * 0.1 + normal_edge_reward_mean * 0.9
                    self.normal_edge_reward_running_var = self.normal_edge_reward_running_var * 0.1 + normal_edge_reward_var * 0.9
                    self.reduce_edge_reward_running_mean = self.reduce_edge_reward_running_mean * 0.1 + reduce_edge_reward_mean * 0.9
                    self.reduce_edge_reward_running_var = self.reduce_edge_reward_running_var * 0.1 + reduce_edge_reward_var * 0.9
                    if not self.args.edge_reward_norm_mean_0:
                        self.normal_edge_reward = (normal_weights.grad.sum(-1) - self.normal_edge_reward_running_mean) / self.normal_edge_reward_running_var
                        self.reduce_edge_reward = (reduce_weights.grad.sum(-1) - self.reduce_edge_reward_running_mean) / self.reduce_edge_reward_running_var
                    else:
                        self.normal_edge_reward = normal_weights.grad.sum(-1) / self.normal_edge_reward_running_var
                        self.reduce_edge_reward = reduce_weights.grad.sum(-1) / self.reduce_edge_reward_running_var
                else:
                    self.normal_edge_reward = normal_weights.grad.sum(-1)
                    self.reduce_edge_reward = reduce_weights.grad.sum(-1)
                self.normal_log_alpha.grad = self.normal_log_alpha.grad.detach() * self.normal_edge_reward.view(-1, 1)
                self.reduce_log_alpha.grad = self.reduce_log_alpha.grad.detach() * self.reduce_edge_reward.view(-1, 1)
                state_dict = self.state_dict()
                child_state_dict = model_child.state_dict()
                for model_child_name, model_child_param in model_child.named_parameters():
                    if model_child_param.grad is not None:
                        state_dict[model_child_name].grad = model_child_param.grad.clone().detach()
                for model_name, model_param in self.named_parameters():
                    if state_dict[model_name].grad is not None:
                        model_param.grad = state_dict[model_name].grad.clone().detach()
                if self.args.prox_policy_opt:
                    self.normal_log_alpha_pre = self.normal_log_alpha.clone().detach()
                    self.reduce_log_alpha_pre = self.reduce_log_alpha.clone().detach()
        if self.snas or self.training:
            if self.args.gen_max_child_flag and not self.training:
                normal_weights = torch.zeros_like(self.normal_log_alpha).scatter_(1, torch.argmax(self.normal_log_alpha, dim=-1).view(-1, 1), 1)
                reduce_weights = torch.zeros_like(self.reduce_log_alpha).scatter_(1, torch.argmax(self.reduce_log_alpha, dim=-1).view(-1, 1), 1)
            s0 = s1 = self.stem(input)
            for i, cell in enumerate(self.cells):
                if self._resample_layer:
                    if cell.reduction:
                        log_alpha = self.reduce_log_alpha
                    else:
                        log_alpha = self.normal_log_alpha
                    weights = self._get_weights(log_alpha)
                elif cell.reduction:
                    log_alpha = self.reduce_log_alpha
                    weights = reduce_weights
                else:
                    log_alpha = self.normal_log_alpha
                    weights = normal_weights
                s0, result = s1, cell(s0, s1, weights, self._drop_path_prob)
                s1 = result[0]
                if i == 2 * self._layers // 3:
                    if self._auxiliary and self.training:
                        logits_aux = self.auxiliary_head(s1)
                op_size = result[1]
                op_flops = result[2]
                op_mac = result[3]
                discrete_prob_1 = F.softmax(log_alpha, dim=-1)
                resource_size_baseline = op_size * discrete_prob_1
                resource_flops_baseline = op_flops * discrete_prob_1
                resource_mac_baseline = op_mac * discrete_prob_1
                clean_size_baseline = resource_size_baseline.sum(-1).clone()
                clean_size_baseline[torch.abs(resource_size_baseline.sum(-1)) < 1] = 1
                clean_flops_baseline = resource_flops_baseline.sum(-1).clone()
                clean_flops_baseline[torch.abs(resource_flops_baseline.sum(-1)) < 1] = 1
                clean_mac_baseline = resource_mac_baseline.sum(-1).clone()
                clean_mac_baseline[torch.abs(resource_mac_baseline.sum(-1)) < 1] = 1
                log_resource_size_baseline = torch.log(clean_size_baseline)
                log_resource_flops_baseline = torch.log(clean_flops_baseline)
                log_resource_mac_baseline = torch.log(clean_mac_baseline)
                resource_size_average = torch.tensor(np.average((op_size.sum(0) / op_size.shape[0]).tolist()).item(), device=op_size.device)
                resource_flops_average = torch.tensor(np.average((op_flops.sum(0) / op_flops.shape[0]).tolist()).item(), device=op_flops.device)
                resource_mac_average = torch.tensor(np.average((op_mac.sum(0) / op_mac.shape[0]).tolist()).item(), device=op_mac.device)
                clean_size_average = resource_size_average.sum(-1).clone()
                clean_size_average[torch.abs(resource_size_average.sum(-1)) < 1] = 1
                clean_flops_average = resource_flops_average.sum(-1).clone()
                clean_flops_average[torch.abs(resource_flops_average.sum(-1)) < 1] = 1
                clean_mac_average = resource_mac_average.sum(-1).clone()
                clean_mac_average[torch.abs(resource_mac_average.sum(-1)) < 1] = 1
                log_resource_size_average = torch.log(clean_size_average)
                log_resource_flops_average = torch.log(clean_flops_average)
                log_resource_mac_average = torch.log(clean_mac_average)
                resource_size = op_size * weights
                resource_flops = op_flops * weights
                resource_mac = op_mac * weights
                clean_size = resource_size.sum(-1).clone()
                clean_flops = resource_flops.sum(-1).clone()
                clean_mac = resource_mac.sum(-1).clone()
                clean_size[torch.abs(resource_size.sum(-1)) < 1] = 1
                clean_flops[torch.abs(resource_flops.sum(-1)) < 1] = 1
                clean_mac[torch.abs(resource_mac.sum(-1)) < 1] = 1
                log_resource_size = torch.log(torch.abs(clean_size))
                log_resource_flops = torch.log(torch.abs(clean_flops))
                log_resource_mac = torch.log(torch.abs(clean_mac))
                resource_size_minus_average = resource_size.sum(-1) - resource_size_average
                resource_flops_minus_average = resource_flops.sum(-1) - resource_flops_average
                resource_mac_minus_average = resource_mac.sum(-1) - resource_mac_average
                log_resource_size_minus_average = log_resource_size - log_resource_size_average
                log_resource_flops_minus_average = log_resource_flops - log_resource_flops_average
                log_resource_mac_minus_average = log_resource_mac - log_resource_mac_average
                if self._method == 'reparametrization':
                    if self._separation == 'all':
                        resource_penalty = ((resource_size * 2 + resource_flops / 4000 + resource_mac / 100) * 0.43).sum(-1)
                        log_resource_penalty = (log_resource_size + log_resource_flops + log_resource_mac) / 3
                    elif self._separation == 'size':
                        resource_penalty = resource_size.sum(-1)
                        log_resource_penalty = log_resource_size
                    elif self._separation == 'flops':
                        resource_penalty = resource_flops.sum(-1)
                        log_resource_penalty = log_resource_flops
                    elif self._separation == 'mac':
                        resource_penalty = resource_mac.sum(-1)
                        log_resource_penalty = log_resource_mac
                    else:
                        resource_penalty = torch.zeros_like(resource_size.sum(-1))
                        log_resource_penalty = resource_penalty
                elif self._method == 'policy_gradient':
                    if self._separation == 'all':
                        if self._minus_baseline:
                            resource_penalty = (resource_size_minus_average * 2 + resource_flops_minus_average / 4000 + resource_mac_minus_average / 100) * 0.43
                            log_resource_penalty = (log_resource_size_minus_average + log_resource_flops_minus_average + log_resource_mac_minus_average) / 3
                        else:
                            resource_penalty = ((resource_size * 2 + resource_flops / 4000 + resource_mac / 100) * 0.43).sum(-1)
                            log_resource_penalty = (log_resource_size + log_resource_flops + log_resource_mac) / 3
                    elif self._separation == 'size':
                        if self._minus_baseline:
                            resource_penalty = resource_size_minus_average
                            log_resource_penalty = log_resource_size_minus_average
                        else:
                            resource_penalty = resource_size.sum(-1)
                            log_resource_penalty = log_resource_size
                    elif self._separation == 'flops':
                        if self._minus_baseline:
                            resource_penalty = resource_flops_minus_average
                            log_resource_penalty = log_resource_flops_minus_average
                        else:
                            resource_penalty = resource_flops.sum(-1)
                            log_resource_penalty = log_resource_flops
                    elif self._separation == 'mac':
                        if self._minus_baseline:
                            resource_penalty = resource_mac_minus_average
                            log_resource_penalty = log_resource_mac_minus_average
                        else:
                            resource_penalty = resource_mac.sum(-1)
                            log_resource_penalty = log_resource_mac
                    else:
                        resource_penalty = torch.zeros_like(resource_size.sum(-1))
                        log_resource_penalty = resource_penalty
                elif self._method == 'discrete':
                    if self._separation == 'all':
                        resource_penalty = ((resource_size_baseline * 2 + resource_flops_baseline / 4000 + resource_mac_baseline / 100) * 0.43).sum(-1)
                        log_resource_penalty = (log_resource_size_baseline + log_resource_flops_baseline + log_resource_mac_baseline) / 3
                    elif self._separation == 'size':
                        resource_penalty = resource_size_baseline.sum(-1)
                        log_resource_penalty = log_resource_size_baseline
                    elif self._separation == 'flops':
                        resource_penalty = resource_flops_baseline.sum(-1)
                        log_resource_penalty = log_resource_flops_baseline
                    elif self._separation == 'mac':
                        resource_penalty = resource_mac_baseline.sum(-1)
                        log_resource_penalty = log_resource_mac_baseline
                    else:
                        resource_penalty = torch.zeros_like(resource_size_baseline.sum(-1))
                        log_resource_penalty = resource_penalty
                else:
                    resource_penalty = torch.zeros_like(resource_size_baseline.sum(-1))
                    log_resource_penalty = resource_penalty
                if self._method == 'policy_gradient':
                    concrete_log_prob = self.logp(log_alpha, weights)
                    resource_penalty = resource_penalty.data
                    log_resource_penalty = log_resource_penalty.data
                    if cell.reduction:
                        total_penalty[7] += (concrete_log_prob * resource_penalty).sum()
                        total_penalty[36] += (concrete_log_prob * log_resource_penalty).sum()
                    else:
                        total_penalty[2] += (concrete_log_prob * resource_penalty).sum()
                        total_penalty[35] += (concrete_log_prob * log_resource_penalty).sum()
                elif self._method == 'reparametrization':
                    if cell.reduction:
                        total_penalty[25] += resource_penalty.sum()
                        total_penalty[38] += log_resource_penalty.sum()
                    else:
                        total_penalty[26] += resource_penalty.sum()
                        total_penalty[37] += log_resource_penalty.sum()
                elif self._method == 'discrete':
                    if cell.reduction:
                        total_penalty[27] += resource_penalty.sum()
                        total_penalty[40] += log_resource_penalty.sum()
                    else:
                        total_penalty[28] += resource_penalty.sum()
                        total_penalty[39] += log_resource_penalty.sum()
                else:
                    total_penalty[-1] += resource_penalty.sum()
                    total_penalty[-2] += resource_penalty.sum()
                concrete_log_prob = self.logp(log_alpha, weights)
                if cell.reduction:
                    total_penalty[8] += resource_size_baseline.sum()
                    total_penalty[9] += resource_flops_baseline.sum()
                    total_penalty[10] += resource_mac_baseline.sum()
                    total_penalty[24] += resource_penalty.sum()
                    total_penalty[59] += log_resource_penalty.sum()
                    total_penalty[32] += resource_size.sum()
                    total_penalty[33] += resource_flops.sum()
                    total_penalty[34] += resource_mac.sum()
                    total_penalty[44] += log_resource_size.sum()
                    total_penalty[45] += log_resource_flops.sum()
                    total_penalty[46] += log_resource_mac.sum()
                    total_penalty[50] += (concrete_log_prob * resource_size.sum(-1)).sum()
                    total_penalty[51] += (concrete_log_prob * resource_flops.sum(-1)).sum()
                    total_penalty[52] += (concrete_log_prob * resource_mac.sum(-1)).sum()
                    total_penalty[56] += (concrete_log_prob * log_resource_size).sum()
                    total_penalty[57] += (concrete_log_prob * log_resource_flops).sum()
                    total_penalty[58] += (concrete_log_prob * log_resource_mac).sum()
                    total_penalty[63] += resource_size_minus_average.sum()
                    total_penalty[64] += resource_flops_minus_average.sum()
                    total_penalty[65] += resource_mac_minus_average.sum()
                    total_penalty[69] += log_resource_size_minus_average.sum()
                    total_penalty[70] += log_resource_flops_minus_average.sum()
                    total_penalty[71] += log_resource_mac_minus_average.sum()
                    op_reduce[0] += op_size
                    op_reduce[1] += op_flops
                    op_reduce[2] += op_mac
                else:
                    total_penalty[3] += resource_size_baseline.sum()
                    total_penalty[5] += resource_flops_baseline.sum()
                    total_penalty[6] += resource_mac_baseline.sum()
                    total_penalty[23] += resource_penalty.sum()
                    total_penalty[29] += resource_size.sum()
                    total_penalty[30] += resource_flops.sum()
                    total_penalty[31] += resource_mac.sum()
                    total_penalty[41] += log_resource_size.sum()
                    total_penalty[42] += log_resource_flops.sum()
                    total_penalty[43] += log_resource_mac.sum()
                    total_penalty[47] += (concrete_log_prob * resource_size.sum(-1)).sum()
                    total_penalty[48] += (concrete_log_prob * resource_flops.sum(-1)).sum()
                    total_penalty[49] += (concrete_log_prob * resource_mac.sum(-1)).sum()
                    total_penalty[53] += (concrete_log_prob * log_resource_size).sum()
                    total_penalty[54] += (concrete_log_prob * log_resource_flops).sum()
                    total_penalty[55] += (concrete_log_prob * log_resource_mac).sum()
                    total_penalty[60] += resource_size_minus_average.sum()
                    total_penalty[61] += resource_flops_minus_average.sum()
                    total_penalty[62] += resource_mac_minus_average.sum()
                    total_penalty[66] += log_resource_size_minus_average.sum()
                    total_penalty[67] += log_resource_flops_minus_average.sum()
                    total_penalty[68] += log_resource_mac_minus_average.sum()
                    op_normal[0] += op_size
                    op_normal[1] += op_flops
                    op_normal[2] += op_mac
        if self.args.snas:
            out = self.global_pooling(s1)
            logits = self.classifier(out.view(out.size(0), -1))
            return logits, logits_aux, total_penalty, op_normal, op_reduce
        else:
            return logits, error_loss, loss_alpha, total_penalty

    def _discrete_prob(self, log_alpha):
        discrete_prob_1 = F.softmax(log_alpha, dim=-1)
        discrete_prob_0 = 1 - discrete_prob_1
        return discrete_prob_0, discrete_prob_1

    def _arch_entropy(self, log_alpha):
        discrete_prob = F.softmax(log_alpha, dim=-1)
        epsilon = 0.0001
        discrete_prob = torch.max(discrete_prob, torch.ones_like(discrete_prob) * epsilon)
        arch_entropy = -(discrete_prob * torch.log(discrete_prob)).sum()
        return arch_entropy

    def _get_categ_mask(self, log_alpha):
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + -(-u.log()).log()) / self._temp)
        return one_hot

    def _get_onehot_mask(self, log_alpha):
        if self.args.random_sample:
            uni = torch.ones_like(log_alpha)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
            one_hot = m.sample()
            return one_hot
        else:
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()

    def _get_weights(self, log_alpha):
        if self.args.dsnas or self.args.random_sample and not self.args.random_sample_fix_temp:
            return self._get_onehot_mask(log_alpha)
        else:
            return self._get_categ_mask(log_alpha)

    def arch_parameters(self):
        return self._arch_parameters

    def load_child_state_dict(self, model_child):
        model_dict = self.state_dict()
        model_child.load_state_dict(model_dict, strict=False)

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.normal_log_alpha, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.reduce_log_alpha, dim=-1).detach().cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
        return genotype

    def genotype_edge_all(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.normal_log_alpha, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.reduce_log_alpha, dim=-1).detach().cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
        return genotype

    def genotype_child(self, normal_weights, reduce_weights):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((self.Primitives[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(normal_weights.detach().cpu().numpy())
        gene_reduce = _parse(reduce_weights.detach().cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
        return genotype


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def broadcast_params(model):
    """ broadcast model parameters """
    for name, p in model.state_dict().items():
        dist.broadcast(p, 0)


class DistModule(torch.nn.Module):

    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)
        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):

        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook


class Conv1x1(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv1x1, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.size = C_in * C_out * kernel_size * kernel_size
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (4 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


class Conv3x3(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv3x3, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.size = C_in * C_out * kernel_size * kernel_size
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (4 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


class Conv5x5(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv5x5, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.size = C_in * C_out * kernel_size * kernel_size
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = y ** 2 * (self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = y ** 2 * (4 * self.C_in + self.C_out) + self.size
            self.flops = self.size * y ** 2
        return self.op(x)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % 4 == 0
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class Shufflenet(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride, bn_affine=True, bn_eps=0.01):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        self.base_mid_channel = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False), nn.BatchNorm2d(outputs, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True)]
        self.branch_main = nn.Sequential(*branch_main)
        if stride == 2:
            branch_proj = [nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False), nn.BatchNorm2d(inp, affine=bn_affine, eps=bn_eps), nn.Conv2d(inp, inp, 1, 1, 0, bias=False), nn.BatchNorm2d(inp, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True)]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, stride, bn_affine=True, bn_eps=1e-05):
        super(Shuffle_Xception, self).__init__()
        assert stride in [1, 2]
        self.base_mid_channel = mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp
        branch_main = [nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp, affine=bn_affine, eps=bn_eps), nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels, affine=bn_affine, eps=bn_eps), nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False), nn.BatchNorm2d(outputs, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True)]
        self.branch_main = nn.Sequential(*branch_main)
        if self.stride == 2:
            branch_proj = [nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp, affine=bn_affine, eps=bn_eps), nn.Conv2d(inp, inp, 1, 1, 0, bias=False), nn.BatchNorm2d(inp, affine=bn_affine, eps=bn_eps), nn.ReLU(inplace=True)]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.LongTensor)


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
    if from_logits:
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs
    masked_indices = None
    num_classes = inputs.size(-1)
    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)
    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)
    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)
    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)
    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())
    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, smooth_eps=self.smooth_eps, smooth_dist=smooth_dist, from_logits=self.from_logits)


def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.0)
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


class BCELoss(nn.BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction, smooth_eps=self.smooth_eps, from_logits=self.from_logits)


class BCEWithLogitsLoss(BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)


class CheckpointModule(nn.Module):

    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, x):
        if self.num_segments > 1:
            return checkpoint_sequential(self.module, self.num_segments, x)
        else:
            return checkpoint(self.module, x)


class SwitchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)
        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


class SyncBNFunc(Function):

    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps = eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True)
            temp = var_in + mean_in ** 2
            if training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2
                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)
            x_hat = (in_data - mean_bn) / (var_bn + ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data
            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data, mean_bn.data, var_bn.data)
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:
            in_data, scale_data, x_hat, mean_bn, var_bn = ctx.saved_tensors
            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat, [0, 2, 3], keepdim=True)
            shiftDiff = torch.sum(grad_outdata, [0, 2, 3], keepdim=True)
            dist.all_reduce(scaleDiff)
            dist.all_reduce(shiftDiff)
            inDiff = scale_data / (var_bn.view(1, C, 1, 1) + ctx.eps).sqrt() * (grad_outdata - 1 / (N * H * W * dist.get_world_size()) * (scaleDiff * x_hat + shiftDiff))
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, None, None, None, None, None


class SyncBatchNorm2d(Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9, last_gamma=False):
        super(SyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, in_data):
        return SyncBNFunc.apply(in_data, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.training)


class SyncSNFunc(Function):

    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, mean_weight, var_weight, running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps = eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True)
            mean_ln = mean_in.mean(1, keepdim=True)
            temp = var_in + mean_in ** 2
            var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
            if training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2
                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)
            softmax = nn.Softmax(0)
            mean_weight = softmax(mean_weight)
            var_weight = softmax(var_weight)
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
            x_hat = (in_data - mean) / (var + ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data
            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data, mean.data, var.data, mean_in.data, var_in.data, mean_ln.data, var_ln.data, mean_bn.data, var_bn.data, mean_weight.data, var_weight.data)
        else:
            raise RuntimeError('SyncSNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:
            in_data, scale_data, x_hat, mean, var, mean_in, var_in, mean_ln, var_ln, mean_bn, var_bn, mean_weight, var_weight = ctx.saved_tensors
            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat, [0, 2, 3], keepdim=True)
            shiftDiff = torch.sum(grad_outdata, [0, 2, 3], keepdim=True)
            x_hatDiff = scale_data * grad_outdata
            meanDiff = -1 / (var.view(N, C) + ctx.eps).sqrt() * torch.sum(x_hatDiff, [2, 3])
            varDiff = -0.5 / (var.view(N, C) + ctx.eps) * torch.sum(x_hatDiff * x_hat, [2, 3])
            term1 = grad_outdata * scale_data / (var.view(N, C, 1, 1) + ctx.eps).sqrt()
            term21 = var_weight[0] * 2 * (in_data.view(N, C, H, W) - mean_in.view(N, C, 1, 1)) / (H * W) * varDiff.view(N, C, 1, 1)
            term22 = var_weight[1] * 2 * (in_data.view(N, C, H, W) - mean_ln.view(N, 1, 1, 1)) / (C * H * W) * torch.sum(varDiff, [1]).view(N, 1, 1, 1)
            term23_tmp = torch.sum(varDiff, [0]).view(1, C, 1, 1)
            dist.all_reduce(term23_tmp)
            term23 = var_weight[2] * 2 * (in_data.view(N, C, H, W) - mean_bn.view(1, C, 1, 1)) / (N * H * W) * term23_tmp / dist.get_world_size()
            term31 = mean_weight[0] * meanDiff.view(N, C, 1, 1) / H / W
            term32 = mean_weight[1] * torch.sum(meanDiff, [1]).view(N, 1, 1, 1) / C / H / W
            term33_tmp = torch.sum(meanDiff, [0]).view(1, C, 1, 1)
            dist.all_reduce(term33_tmp)
            term33 = mean_weight[2] * term33_tmp / N / H / W / dist.get_world_size()
            inDiff = term1 + term21 + term22 + term23 + term31 + term32 + term33
            mw1_diff = torch.sum(meanDiff * mean_in.view(N, C))
            mw2_diff = torch.sum(meanDiff * mean_ln.view(N, 1))
            mw3_diff = torch.sum(meanDiff * mean_bn.view(1, C))
            dist.all_reduce(mw1_diff)
            dist.all_reduce(mw2_diff)
            dist.all_reduce(mw3_diff)
            vw1_diff = torch.sum(varDiff * var_in.view(N, C))
            vw2_diff = torch.sum(varDiff * var_ln.view(N, 1))
            vw3_diff = torch.sum(varDiff * var_bn.view(1, C))
            dist.all_reduce(vw1_diff)
            dist.all_reduce(vw2_diff)
            dist.all_reduce(vw3_diff)
            mean_weight_Diff = mean_weight
            var_weight_Diff = var_weight
            mean_weight_Diff[0] = mean_weight[0] * (mw1_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            mean_weight_Diff[1] = mean_weight[1] * (mw2_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            mean_weight_Diff[2] = mean_weight[2] * (mw3_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            var_weight_Diff[0] = var_weight[0] * (vw1_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[1] = var_weight[1] * (vw2_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[2] = var_weight[2] * (vw3_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, mean_weight_Diff, var_weight_Diff, None, None, None, None, None


class SyncSwitchableNorm2d(Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9, last_gamma=False):
        super(SyncSwitchableNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.mean_weight = Parameter(torch.ones(3))
        self.var_weight = Parameter(torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, in_data):
        return SyncSNFunc.apply(in_data, self.weight, self.bias, self.mean_weight, self.var_weight, self.running_mean, self.running_var, self.eps, self.momentum, self.training)


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, input_size=224, n_class=1000, args=None, architecture=None, channels_scales=None):
        super(ShuffleNetV2_OneShot, self).__init__()
        assert input_size % 32 == 0
        assert architecture is not None and channels_scales is not None
        self.arch = architecture
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        self.args = args
        self.bn_affine = args.bn_affine
        self.bn_eps = args.bn_eps
        self.num_blocks = 4
        self.device = torch.device('cuda')
        self.log_alpha = torch.nn.Parameter(torch.zeros(sum(self.stage_repeats), self.num_blocks).normal_(self.args.loc_mean, self.args.loc_std).requires_grad_())
        self._arch_parameters = [self.log_alpha]
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel, eps=self.bn_eps), nn.ReLU(inplace=True))
        self.features = nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1
                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels * channels_scales[archIndex])
                pos = self.arch[archIndex]
                archIndex += 1
                blocks = nn.ModuleList()
                if pos == 0:
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(None)
                elif pos == 1:
                    blocks.append(None)
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                    blocks.append(None)
                elif pos == 2:
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                    blocks.append(None)
                elif pos == 3:
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(None)
                    blocks.append(Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride, bn_affine=self.bn_affine, bn_eps=self.bn_eps))
                input_channel = output_channel
                self.features += [blocks]
        self.conv_last = nn.Sequential(nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False), nn.BatchNorm2d(self.stage_out_channels[-1], eps=self.bn_eps), nn.ReLU(inplace=True))
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))

    def forward(self, x, target=None, criterion=None):
        error_loss = 0
        loss_alpha = 0
        x = self.first_conv(x)
        for i, block in enumerate(self.features):
            pos = self.arch[i]
            x = self.features[i][pos](x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CheckpointModule,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1x1,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3x3,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv5x5,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistModule,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardZero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwitchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SwitchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwitchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SNAS_Series_SNAS_Series(_paritybench_base):
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

