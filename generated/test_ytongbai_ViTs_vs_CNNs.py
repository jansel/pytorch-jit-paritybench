import sys
_module = sys.modules[__name__]
del sys
autoattack = _module
autoattack = _module
autopgd_base = _module
autopgd_pt = _module
autopgd_tf = _module
eval = _module
eval_tf1 = _module
eval_tf2 = _module
resnet = _module
fab_base = _module
fab_projections = _module
fab_pt = _module
fab_tf = _module
other_utils = _module
square = _module
utils_tf = _module
utils_tf2 = _module
setup = _module
attacker = _module
ghost_bn = _module
ghost_bn_old = _module
misc = _module
models = _module
resnet_gbn = _module
resnet_gbn_gelu_4096 = _module
test_autoattack = _module
vision_transformer = _module
datasets = _module
engine = _module
engine_accum = _module
engine_clean = _module
global_val = _module
imagenet_robust = _module
main_adv_deit = _module
main_adv_res = _module
main_clean = _module
advresnet = _module
advresnet_gbn = _module
advresnet_gbn_gelu = _module
affine = _module
attacker = _module
ghost_bn = _module
losses = _module
models = _module
models_clean = _module
resnet_gbn = _module
vision_transformer = _module
run_with_submitit = _module
samplers = _module
scaler = _module
util = _module
imagenet_a = _module
logger = _module
lr_scheduler = _module
misc = _module
progress = _module
bar = _module
counter = _module
helpers = _module
spinner = _module
test_progress = _module
visualize = _module
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


from torch.autograd import Variable


import numpy as np


import time


import math


import random


import torchvision.datasets as datasets


import torch.utils.data as data


import torchvision.transforms as transforms


import tensorflow as tf


from torch.nn import functional as F


import typing


from torch import Tensor


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


from functools import partial


import functools


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


from torch.utils.data import Dataset


from typing import Iterable


from typing import Optional


from torch.optim.lr_scheduler import _LRScheduler


import torch.distributed as dist


import matplotlib.pyplot as plt


import torchvision


from collections import defaultdict


from collections import deque


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class GhostBN2D(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=1, affine=True, sync_stats=False, **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features * virtual2actual_batch_size_ratio, *args, **kwargs, affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        del self.proxy_bn.weight
        del self.proxy_bn.bias
        self.reset_parameters()
        self.eval_use_different_stats = False

    def reset_parameters(self) ->None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) ->typing.Tuple[Tensor, Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {(False): lambda x: x[0], (True): lambda x: torch.mean(x, dim=0)}[self.sync_stats]
            return tuple(select_fun(var.reshape(self.virtual2actual_batch_size_ratio, self.num_features)) for var in [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def _prepare_fake_weight_bias(self):
        if not self.affine:
            self.proxy_bn.weight = None
            self.proxy_bn.bias = None
        _fake_weight, _fake_bias = [var.unsqueeze(0).expand(self.virtual2actual_batch_size_ratio, self.num_features).reshape(-1) for var in [self.weight, self.bias]]
        self.proxy_bn.weight = _fake_weight
        self.proxy_bn.bias = _fake_bias

    def forward(self, input: Tensor) ->Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = self.proxy_bn.running_mean is None and self.proxy_bn.running_var is None
        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(int(n / self.virtual2actual_batch_size_ratio), self.virtual2actual_batch_size_ratio * c, h, w)
            self._prepare_fake_weight_bias()
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)
            return proxy_output
        else:
            running_mean, running_var = self.get_actual_running_stats()
            return F.batch_norm(input, running_mean, running_var, self.weight, self.bias, bn_training, 0.0, self.proxy_bn.eps)


class GhostBN2D_Old(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features * virtual2actual_batch_size_ratio, *args, **kwargs, affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eval_use_different_stats = False

    def reset_parameters(self) ->None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) ->typing.Tuple[Tensor, Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {(False): lambda x: x[0], (True): lambda x: torch.mean(x, dim=0)}[self.sync_stats]
            return tuple(select_fun(var.reshape(self.virtual2actual_batch_size_ratio, self.num_features)) for var in [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def forward(self, input: Tensor) ->Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = self.proxy_bn.running_mean is None and self.proxy_bn.running_var is None
        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(int(n / self.virtual2actual_batch_size_ratio), self.virtual2actual_batch_size_ratio * c, h, w)
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)
            if self.affine:
                weight = self.weight
                bias = self.bias
                weight = weight.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                return proxy_output * weight + bias
            else:
                return proxy_output
        else:
            running_mean, running_var = self.get_actual_running_stats()
            return F.batch_norm(input, running_mean, running_var, self.weight, self.bias, bn_training, 0.0, self.proxy_bn.eps)


class BGN(nn.Module):

    def __init__(self, num_channels, num_groups=2, eps=1e-05, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.groupnorm = nn.GroupNorm(num_channels=self.num_channels, num_groups=self.num_groups, eps=self.eps)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.groupnorm(x)
        x = x.permute(2, 0, 1)
        return x


class GN(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.groupnorm = nn.GroupNorm(num_channels=self.num_channels, num_groups=self.num_groups, eps=self.eps)

    def forward(self, x):
        n = x.shape[0]
        p = x.shape[1]
        c = x.shape[2]
        x = x.reshape(-1, c)
        x = self.groupnorm(x)
        x = x.reshape(n, p, c)
        return x


class BN(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.batchnorm = nn.BatchNorm1d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1)
        return x


class Affine(nn.Module):

    def __init__(self, width, *args, k=1, **kwargs):
        super(Affine, self).__init__()
        self.bnconv = nn.Conv2d(width, width, k, padding=(k - 1) // 2, groups=width, bias=True)

    def forward(self, x):
        return self.bnconv(x)


class NoOpAttacker:

    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


def to_bn(m, status):
    if hasattr(m, 'bn_type'):
        m.bn_type = status


to_0 = partial(to_bn, status='bn0')


to_1 = partial(to_bn, status='bn1')


to_2 = partial(to_bn, status='bn2')


to_3 = partial(to_bn, status='bn3')


def eval_use_different_stats(model, val=False):

    def aux(m):
        if isinstance(m, GhostBN2D):
            m.eval_use_different_stats = val
    model.apply(aux)


to_adv = functools.partial(eval_use_different_stats, val=True)


to_clean = functools.partial(eval_use_different_stats, val=False)


class AdvResNet(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, attacker=NoOpAttacker()):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)
        self.attacker = attacker
        self.mix = False
        self.sing = False
        self.mixup_fn = False
        self.bn_num = 0

    def set_mixup_fn(self, mixup):
        self.mixup_fn = mixup

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_bn_num(self, bn_num):
        self.bn_num = bn_num

    def set_mix(self, mix):
        self.mix = mix

    def set_sing(self, sing):
        self.sing = sing

    def forward(self, x, labels):
        if self.sing:
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv)
                if global_val.gbn_forward_time == 4:
                    if self.bn_num == 0:
                        self.apply(to_0)
                    elif self.bn_num == 1:
                        self.apply(to_1)
                    elif self.bn_num == 2:
                        self.apply(to_2)
                    elif self.bn_num == 3:
                        self.apply(to_3)
                elif global_val.gbn_forward_time == 8:
                    if self.bn_num == 0:
                        self.apply(to_0)
                    elif self.bn_num == 1:
                        self.apply(to_1)
                    elif self.bn_num == 2:
                        self.apply(to_2)
                    elif self.bn_num == 3:
                        self.apply(to_3)
                    elif self.bn_num == 4:
                        self.apply(to_4)
                    elif self.bn_num == 5:
                        self.apply(to_5)
                    elif self.bn_num == 6:
                        self.apply(to_6)
                    elif self.bn_num == 7:
                        self.apply(to_7)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, True, self.mixup_fn)
                    images = aux_images
                    targets = labels
                self.train()
                self.apply(to_clean)
                return self._forward_impl(images), targets
            else:
                self.apply(to_0)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, False, False)
                    images = aux_images
                    targets = labels
                return self._forward_impl(images), targets


class GhostBN2D_ADV(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features * virtual2actual_batch_size_ratio, *args, **kwargs, affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eval_use_different_stats = False

    def reset_parameters(self) ->None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) ->typing.Tuple[Tensor, Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {(False): lambda x: x[0], (True): lambda x: torch.mean(x, dim=0)}[self.sync_stats]
            return tuple(select_fun(var.reshape(self.virtual2actual_batch_size_ratio, self.num_features)) for var in [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def forward(self, input: Tensor) ->Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = self.proxy_bn.running_mean is None and self.proxy_bn.running_var is None
        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(int(n / self.virtual2actual_batch_size_ratio), self.virtual2actual_batch_size_ratio * c, h, w)
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)
            if self.affine:
                weight = self.weight
                bias = self.bias
                weight = weight.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                return proxy_output * weight + bias
            else:
                return proxy_output
        else:
            running_mean, running_var = self.get_actual_running_stats()
            return F.batch_norm(input, running_mean, running_var, self.weight, self.bias, bn_training, 0.0, self.proxy_bn.eps)


class FourBN(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super(FourBN, self).__init__()
        virtual2actual_batch_size_ratio = global_val.ratio_
        self.bn0 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)

    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        input = self.aff(input)
        return input


class EightBN(nn.Module):

    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super(EightBN, self).__init__()
        virtual2actual_batch_size_ratio = global_val.ratio_
        self.bn0 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn4 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn5 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn6 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn7 = GhostBN2D_ADV(*args, num_features=num_features, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)

    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        elif self.bn_type == 'bn4':
            input = self.bn4(input)
        elif self.bn_type == 'bn5':
            input = self.bn5(input)
        elif self.bn_type == 'bn6':
            input = self.bn6(input)
        elif self.bn_type == 'bn7':
            input = self.bn7(input)
        input = self.aff(input)
        return input


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.apply(self._init_weights_2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_2(self, m):
        if isinstance(m, Block):
            nn.init.constant_(m.attn.proj.weight, 0)
            nn.init.constant_(m.mlp.fc2.weight, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def _forward_impl(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class SingLN(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-06, elementwise_affine=True):
        super(SingLN, self).__init__(normalized_shape, eps, elementwise_affine)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = super(SingLN, self).forward(input)
        elif self.batch_type == 'clean':
            input = super(SingLN, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            input = super(SingLN, self).forward(input)
        return input


class DistilledVisionTransformer(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_adv_status = partial(to_status, status='adv')


to_clean_status = partial(to_status, status='clean')


to_mix_status = partial(to_status, status='mix')


class AdvVisionTransformer(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, attacker=NoOpAttacker()):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        self.attacker = attacker
        self.mix = False
        self.sing = False
        self.mixup_fn = False

    def set_mixup_fn(self, mixup):
        self.mixup_fn = mixup

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_mix(self, mix):
        self.mix = mix

    def set_sing(self, sing):
        self.sing = sing

    def forward(self, x, labels):
        if not self.sing:
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv_status)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, False)
                    images = torch.cat([x, aux_images], dim=0)
                    targets = torch.cat([labels, labels], dim=0)
                self.train()
                if self.mix:
                    self.apply(to_mix_status)
                    if len(labels.shape) == 2:
                        return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2, input_len, -1).transpose(1, 0)
                    else:
                        return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2, input_len).transpose(1, 0)
                else:
                    self.apply(to_clean_status)
                    return self._forward_impl(images), targets
            else:
                images = x
                targets = labels
                return self._forward_impl(images), targets
        else:
            training = self.training
            input_len = len(x)
            if training:
                self.eval()
                self.apply(to_adv_status)
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, True, self.mixup_fn)
                    images = aux_images
                    targets = labels
                self.train()
                self.apply(to_clean_status)
                return self._forward_impl(images), targets
            else:
                if isinstance(self.attacker, NoOpAttacker):
                    images = x
                    targets = labels
                else:
                    aux_images, _ = self.attacker.attack(x, labels, self._forward_impl, True, False, False)
                    images = aux_images
                    targets = labels
                return self._forward_impl(images), targets


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        if outputs_kd is None:
            raise ValueError('When knowledge distillation is enabled, the model is expected to return a Tuple[Tensor, Tensor] with the output of the class_token and the dist_token')
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(F.log_softmax(outputs_kd / T, dim=1), F.log_softmax(teacher_outputs / T, dim=1), reduction='sum', log_target=True) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Affine,
     lambda: ([], {'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BGN,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GN,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GhostBN2D,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GhostBN2D_ADV,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GhostBN2D_Old,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingLN,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftTargetCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ytongbai_ViTs_vs_CNNs(_paritybench_base):
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

