import sys
_module = sys.modules[__name__]
del sys
custom_writer = _module
DistributedProxySampler = _module
datasets = _module
randaugment = _module
data_utils = _module
dataset = _module
ssl_dataset = _module
nets = _module
resnet50 = _module
wrn = _module
wrn_var = _module
npmatch = _module
np_head = _module
npmatch = _module
npmatch_utils = _module
npmatch = _module
train_utils = _module
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


from typing import Sequence


from typing import Union


from typing import Tuple


import numpy as np


import torch


import math


from torch.utils.data.distributed import DistributedSampler


import random


import torch.nn.functional as F


import torchvision


from torchvision import datasets


from torch.utils.data import sampler


from torch.utils.data import DataLoader


from torch.utils.data.sampler import BatchSampler


import torch.distributed as dist


from torchvision import transforms


from torch.utils.data import Dataset


import copy


from collections import Counter


from torch import Tensor


import torch.nn as nn


from typing import Type


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


import logging


import pandas as pd


import torchvision.models as models


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from sklearn.metrics import *


from copy import deepcopy


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


from torch.optim.lr_scheduler import LambdaLR


import time


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


def conv1x1(in_planes: int, out_planes: int, stride: int=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
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

    def forward(self, x: Tensor) ->Tensor:
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


class ResNet50(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]]=Bottleneck, layers: List[int]=[3, 4, 6, 3], n_class: int=1000, zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
        super(ResNet50, self).__init__()
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

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1, dilate: bool=False) ->nn.Sequential:
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
        out = torch.flatten(x, 1)
        return out

    def forward(self, x):
        return self._forward_impl(x)


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.channels = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return out


class WideResNetVar(nn.Module):

    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNetVar, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.block4 = NetworkBlock(n, channels[3], channels[4], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.channels = channels[4]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return out


class MLP(nn.Module):

    def __init__(self, layer_sizes=[512, 512], last_act=False):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential()
        if last_act:
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(name='L{:d}'.format(i), module=nn.Linear(in_size, out_size, bias=True))
                self.MLP.add_module(name='A{:d}'.format(i), module=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(name='L{:d}'.format(i), module=nn.Linear(in_size, out_size, bias=True))
                if i < len(layer_sizes[:-1]) - 1:
                    self.MLP.add_module(name='A{:d}'.format(i), module=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.MLP(x)
        return x


class MLP_Decoder(nn.Module):

    def __init__(self, layer_sizes=[512, 512], last_act=False):
        super(MLP_Decoder, self).__init__()
        self.MLP = nn.Sequential()
        if last_act:
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(name='L{:d}'.format(i), module=nn.Linear(in_size, out_size, bias=True))
                self.MLP.add_module(name='A{:d}'.format(i), module=nn.ReLU(inplace=True))
        else:
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(name='L{:d}'.format(i), module=nn.Linear(in_size, out_size, bias=True))
                if i < len(layer_sizes[:-1]) - 1:
                    self.MLP.add_module(name='A{:d}'.format(i), module=nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.MLP(x)
        return x


class NP_HEAD(nn.Module):

    def __init__(self, input_dim, latent_dim, num_classes=10, memory_max_length=2560):
        super(NP_HEAD, self).__init__()
        self.latent_path = MLP(layer_sizes=[input_dim + num_classes, latent_dim, latent_dim], last_act=True)
        self.deterministic_path = MLP(layer_sizes=[input_dim + num_classes, latent_dim, latent_dim], last_act=True)
        self.mean_net = MLP(layer_sizes=[latent_dim, latent_dim, latent_dim])
        self.log_var_net = MLP(layer_sizes=[latent_dim, latent_dim, latent_dim])
        self.num_classes = num_classes
        self.fc_decoder = MLP(layer_sizes=[2 * latent_dim + input_dim, input_dim, input_dim], last_act=True)
        self.classifier = nn.Linear(input_dim, num_classes, bias=True)
        self.memory_max_length = memory_max_length
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x_in, deterministic_memory, latent_memory, forward_times=10, phase_train=True, x_context_in=None, labels_in=None, labels_context_in=None, update_deterministic_memory=True):
        if phase_train:
            x_combine_deterministic_input = torch.cat((x_context_in, labels_context_in), dim=-1)
            x_representation_deterministic = self.deterministic_path(x_combine_deterministic_input)
            x_combine_latent_input = torch.cat((x_in, labels_in), dim=-1)
            x_representation_latent = self.latent_path(x_combine_latent_input)
            mean_x = self.mean_net(x_representation_latent.mean(0))
            log_var_x = self.log_var_net(x_representation_latent.mean(0))
            sigma_x = 0.1 + 0.9 * F.softplus(log_var_x)
            latent_z_target = None
            B = x_in.size(0)
            for i in range(0, forward_times):
                z = self.reparameterize(mean_x, sigma_x)
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            x_target_in_expand = x_in.unsqueeze(0).expand(forward_times, -1, -1)
            context_representation_deterministic_expand = x_representation_deterministic.mean(0).unsqueeze(0).unsqueeze(1).expand(forward_times, B, -1)
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
            if update_deterministic_memory:
                deterministic_memory = torch.cat((deterministic_memory, x_representation_deterministic.detach()), dim=0)
                if deterministic_memory.size(0) > self.memory_max_length:
                    Diff = deterministic_memory.size(0) - self.memory_max_length
                    deterministic_memory = deterministic_memory[Diff:, :]
            latent_memory = torch.cat((latent_memory, x_representation_latent.detach()), dim=0)
            if latent_memory.size(0) > self.memory_max_length:
                Diff = latent_memory.size(0) - self.memory_max_length
                latent_memory = latent_memory[Diff:, :]
            decoder_input_cat = torch.cat((latent_z_target_expand, x_target_in_expand, context_representation_deterministic_expand), dim=-1)
            T, B, D = decoder_input_cat.size()
            output_function = self.fc_decoder(decoder_input_cat)
            output = self.classifier(output_function)
            return output, mean_x, sigma_x, deterministic_memory, latent_memory
        else:
            mean = self.mean_net(latent_memory.mean(0))
            log_var = self.log_var_net(latent_memory.mean(0))
            sigma = 0.1 + 0.9 * F.softplus(log_var)
            latent_z_target = None
            B = x_in.size(0)
            for i in range(0, forward_times):
                z = self.reparameterize(mean, sigma)
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            x_target_in_expand = x_in.unsqueeze(0).expand(forward_times, -1, -1)
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
            context_representation_deterministic_expand = deterministic_memory.mean(0).unsqueeze(0).unsqueeze(1).expand(forward_times, B, -1)
            decoder_input_cat = torch.cat((latent_z_target_expand, x_target_in_expand, context_representation_deterministic_expand), dim=-1)
            T, B, D = decoder_input_cat.size()
            output_function = self.fc_decoder(decoder_input_cat)
            output = self.classifier(output_function)
            return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (WideResNet,
     lambda: ([], {'first_stride': 1, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (WideResNetVar,
     lambda: ([], {'first_stride': 1, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_Jianf_Wang_NP_Match(_paritybench_base):
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

