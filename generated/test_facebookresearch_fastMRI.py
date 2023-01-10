import sys
_module = sys.modules[__name__]
del sys
fastmri = _module
args = _module
base_trainer = _module
checkpointing_mixin = _module
common = _module
evaluate = _module
image_grid = _module
subsample = _module
test_subsample = _module
utils = _module
data = _module
mri_data = _module
test_transforms = _module
transforms = _module
volume_sampler = _module
distributed_mixin = _module
learning_rate_mixin = _module
logging_mixin = _module
model = _module
resnet_r1 = _module
resnet_r1_prev = _module
resnet_r1_simple = _module
resnet_r1_wide = _module
torchvision_resnet = _module
unpooled_resnet = _module
public_unet = _module
var_net = _module
optimizer = _module
adversary_mixin = _module
test_adversary = _module
parameter_group_mixin = _module
run = _module
spawn_dist = _module
ssim_loss_mixin = _module
trainer = _module
training_loop_mixin = _module
transform_mixin = _module
kspace = _module
orientation = _module
var_net_trainer = _module
visualization_mixin = _module
pretrain = _module
pretrain_dev = _module
train = _module
train_dev = _module
coil_combine = _module
mri_data = _module
subsample = _module
transforms = _module
volume_sampler = _module
fftc = _module
losses = _module
math = _module
models = _module
adaptive_varnet = _module
policy = _module
unet = _module
varnet = _module
pl_modules = _module
data_module = _module
mri_module = _module
unet_module = _module
varnet_module = _module
RunInference = _module
fastmri_examples = _module
eval_pretrained_adaptive_varnet = _module
adaptive_varnet_module = _module
metrics = _module
varnet_module = _module
train_adaptive_varnet_demo = _module
util = _module
cs = _module
run_bart = _module
run_pretrained_unet_inference = _module
train_unet_demo = _module
unet_brain_leaderboard = _module
unet_knee_mc_leaderboard = _module
unet_knee_sc_leaderboard = _module
run_pretrained_varnet_inference = _module
train_varnet_demo = _module
varnet_brain_leaderboard = _module
varnet_knee_leaderboard = _module
zero_filled = _module
run_zero_filled = _module
setup = _module
tests = _module
conftest = _module
create_temp_data = _module
test_data = _module
test_integrations = _module
test_math = _module
test_models = _module
test_modules = _module

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


import logging


import math


import random


import functools


from collections import OrderedDict


from collections import namedtuple


import torch


from torch import nn


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


import time


import numpy as np


from torch.nn import functional as F


from torch.utils.data import Dataset


import re


from collections import defaultdict


from scipy.sparse.linalg import svds


from scipy.linalg import svd


import collections


from torch.utils.data import Sampler


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.autograd import Variable


import torch.utils.data


import torch.utils.data.distributed


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


import torch.optim


from torch import autograd


import torchvision


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import NamedTuple


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from warnings import warn


import pandas as pd


import torch.fft


from torch.autograd import Function


from torch import optim


from torchvision import utils


import torch.utils.checkpoint


import matplotlib.pyplot as plt


def actvn(x):
    return F.relu(x)


kernel_size = 5


class ResnetBlock(nn.Module):

    def __init__(self, fin, fout, fhidden=None):
        super().__init__()
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.norm_0 = nn.GroupNorm(self.fin // 32, self.fin)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.norm_1 = nn.GroupNorm(self.fhidden // 32, self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.norm_0(x)))
        dx = self.conv_1(actvn(self.norm_1(dx)))
        out = x_s + dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class Discriminator(nn.Module):
    """
        Known to work well as a GAN discriminator
        
    """

    def __init__(self, num_classes=1, args=None):
        super().__init__()
        nf = self.nf = 192
        nlayers = 1
        self.nf0 = nf * 2 ** nlayers
        blocks = [ResnetBlock(nf, nf), ResnetBlock(nf, nf), ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = nf * 2 ** i
            nf1 = nf * 2 ** (i + 1)
            blocks += [nn.AvgPool2d(2, stride=2, padding=0), ResnetBlock(nf0, nf1), ResnetBlock(nf1, nf1), ResnetBlock(nf1, nf1)]
        self.conv_img = nn.Conv2d(3, 1 * nf, kernel_size=kernel_size, padding=kernel_size // 2)
        self.resnet = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.nf0, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        out = self.conv_img(out)
        out = self.resnet(out)
        out = self.pool(out)
        out = out.view(batch_size, self.nf0)
        out = self.fc(actvn(out))
        return out


class SimpleDiscriminator(nn.Module):
    """
        Known to work well as a GAN discriminator
        
    """

    def __init__(self, num_classes=1, args=None):
        super().__init__()
        nf = self.nf = 128
        nlayers = 0
        self.nf0 = nf
        blocks = [ResnetBlock(nf, nf), ResnetBlock(nf, nf)]
        self.conv_img = nn.Conv2d(3, 1 * nf, kernel_size=kernel_size, padding=kernel_size // 2)
        self.resnet = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.nf0, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        out = self.conv_img(out)
        out = self.resnet(out)
        out = self.pool(out)
        out = out.view(batch_size, self.nf0)
        out = self.fc(actvn(out))
        return out


class WideDiscriminator(nn.Module):
    """
        Known to work well as a GAN discriminator
        
    """

    def __init__(self, num_classes=1, args=None):
        super().__init__()
        nf = self.nf = 128
        nlayers = 1
        self.nf0 = nf * 2 ** nlayers
        blocks = [ResnetBlock(nf, nf), ResnetBlock(nf, nf), ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = nf * 2 ** i
            nf1 = nf * 2 ** (i + 1)
            blocks += [nn.MaxPool2d(4, stride=4, padding=0), ResnetBlock(nf0, nf1), ResnetBlock(nf1, nf1), ResnetBlock(nf1, nf1)]
        self.conv_img = nn.Conv2d(3, 1 * nf, kernel_size=kernel_size, padding=kernel_size // 2)
        self.resnet = nn.Sequential(*blocks)
        self.pool_max = nn.MaxPool2d(4, stride=4, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.nf0, num_classes)
        self.norm = nn.InstanceNorm2d(3, affine=False, eps=0.0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        out = self.norm(out)
        out = self.conv_img(out)
        out = self.resnet(out)
        out = self.pool_max(out)
        out = self.pool(out)
        out = out.view(batch_size, self.nf0)
        out = self.fc(actvn(out))
        return out


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
            norm_layer = lambda f: nn.GroupNorm(f // 32, f)
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Resnet50(ResNet):

    def __init__(self, num_classes=1, args=None):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)


class UnpooledResnet50(ResNet):

    def __init__(self, num_classes=1, args=None):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_chans), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(drop_prob), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_chans), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(drop_prob))

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, variant=None, kernel_size=3, padding=1, dilation=1, groups=1):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob, variant=variant, ks=kernel_size, pad=padding, dil=dilation, num_group=groups)]
        self.conv2 = nn.Sequential(nn.Conv2d(ch, ch // 2, kernel_size=1), nn.Conv2d(ch // 2, out_chans, kernel_size=1), nn.Conv2d(out_chans, out_chans, kernel_size=1))

    def forward(self, input, *args):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        output = self.conv(output)
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = downsample_layer.shape[-2], downsample_layer.shape[-1]
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        return self.conv2(output)


class Push(nn.Module):
    pass


class Pop(nn.Module):

    def __init__(self, method):
        super().__init__()
        self.method = method


def conv(in_channels, out_channels, transpose=False, kernel_size=3):
    if transpose:
        yield nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False)
    else:
        yield nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False)
    yield nn.InstanceNorm2d(out_channels)
    yield nn.LeakyReLU(0.2, True)


class UnetModel2(nn.Module):

    def __init__(self, in_chans, out_chans, chans):
        super().__init__()
        c = chans
        self.layers = nn.ModuleList([*conv(in_chans, 1 * c), *conv(1 * c, 1 * c), Push(), nn.AvgPool2d(2, 2), *conv(1 * c, 2 * c), *conv(2 * c, 2 * c), Push(), nn.AvgPool2d(2, 2), *conv(2 * c, 4 * c), *conv(4 * c, 4 * c), Push(), nn.AvgPool2d(2, 2), *conv(4 * c, 8 * c), *conv(8 * c, 8 * c), Push(), nn.AvgPool2d(2, 2), *conv(8 * c, 16 * c), *conv(16 * c, 16 * c), *conv(16 * c, 8 * c, transpose=True), Pop(), *conv(16 * c, 8 * c), *conv(8 * c, 8 * c), *conv(8 * c, 4 * c, transpose=True), Pop(), *conv(8 * c, 4 * c), *conv(4 * c, 4 * c), *conv(4 * c, 2 * c, transpose=True), Pop(), *conv(4 * c, 2 * c), *conv(2 * c, 2 * c), *conv(2 * c, 1 * c, transpose=True), Pop(), *conv(2 * c, 1 * c), *conv(1 * c, 1 * c), nn.Conv2d(1 * c, out_chans, 1)])

    def forward(self, input, *args):
        self.stack = []
        x = input
        for lyr in self.layers:
            if isinstance(lyr, Push):
                self.stack.append(x)
            elif isinstance(lyr, Pop):
                x = torch.cat([x, self.stack.pop()], dim=1)
            else:
                x = lyr(x)
        return x


def complex_abs2(x):
    assert x.shape[-1] == 2
    return x[..., 0] ** 2 + x[..., 1] ** 2


class Abs(nn.Module):

    def forward(self, x):
        return complex_abs2(x).sqrt()


def mask_center(x, num_lf):
    b, c, h, w, two = x.shape
    assert b == 1
    pad = (w - num_lf + 1) // 2
    y = torch.zeros_like(x)
    y[:, :, :, pad:pad + num_lf] = x[:, :, :, pad:pad + num_lf]
    return y


class MaskCenter(nn.Module):

    def forward(self, x, input):
        return mask_center(x, input['num_lf'].item())


def rss(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return complex_abs2(x).sum(dim=1, keepdim=True).sqrt()


class RSS(nn.Module):

    def forward(self, x):
        return rss(x)


class dRSS(nn.Module):

    def forward(self, x):
        return x / rss(x).unsqueeze(-1)


class Fm2Batch(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b, c, *other = x.shape
        x = x.contiguous().view(b * c, 1, *other)
        x = self.model(x)
        x = x.view(b, c, *other)
        return x


class Complex2Fm(nn.Module):

    def forward(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)


class Fm2Complex(nn.Module):

    def forward(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return x.view(b, 2, c2 // 2, h, w).permute(0, 2, 3, 4, 1)


class Polar(nn.Module):

    def forward(self, x):
        r = complex_abs2(x).sqrt()
        phi = torch.atan2(x[..., 1], x[..., 0])
        return torch.stack((r, phi), dim=-1)


class Cartesian(nn.Module):

    def forward(self, x):
        r, phi = x[..., 0], x[..., 1]
        return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)


def roll(x, shift, dim):
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim // 2) for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [(x.shape[i] // 2) for i in dim]
    return roll(x, shift, dim)


def ft(x):
    assert x.shape[-1] == 2
    x = fftshift(x, dim=(-3, -2))
    x = torch.fft(x, 2, normalized=True)
    x = fftshift(x, dim=(-3, -2))
    return x


class FT(nn.Module):

    def forward(self, x):
        return ft(x)


def ift(x):
    assert x.shape[-1] == 2
    x = fftshift(x, dim=(-3, -2))
    x = torch.ifft(x, 2, normalized=True)
    x = fftshift(x, dim=(-3, -2))
    return x


class IFT(nn.Module):

    def forward(self, x):
        return ift(x)


def norm(x, model, norm_type, norm_mean, norm_std):
    b, c, h, w = x.shape
    mean, std = 0, 1
    if norm_type == 'layer':
        x = x.contiguous().view(b, c * h * w)
        if norm_mean:
            mean = x.mean(dim=1).view(b, 1, 1, 1)
        if norm_std:
            std = x.std(dim=1).view(b, 1, 1, 1)
    elif norm_type == 'instance':
        x = x.contiguous().view(b, c, h * w)
        if norm_mean:
            mean = x.mean(dim=2).view(b, c, 1, 1)
        if norm_std:
            std = x.std(dim=2).view(b, c, 1, 1)
    elif norm_type == 'group':
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        if norm_mean:
            mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        if norm_std:
            std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
    x = x.view(b, c, h, w)
    x = (x - mean) / std
    x = model(x)
    x = x * std + mean
    return x


class Norm(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return norm(x, self.model, args.norm_type, args.norm_mean, args.norm_std)


def dc(x, mask, kspace):
    return torch.where(mask, kspace, x)


class DC(nn.Module):

    def forward(self, x, input):
        return dc(x, input['mask'], input['kspace'])


def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


def sens_expand(x, sens_maps):
    return complex_mul(x, sens_maps)


class SensExpand(nn.Module):

    def forward(self, x, input):
        return sens_expand(x, input['sens_maps'])


def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def sens_reduce(x, sens_maps):
    return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)


class SensReduce(nn.Module):

    def forward(self, x, input):
        return sens_reduce(x, input['sens_maps'])


class GRAPPA(nn.Module):

    def __init__(self, acceleration):
        super().__init__()
        self.acceleration = acceleration

    def forward(self, x, input):
        return T.apply_grappa(x, input[f'grappa_{self.acceleration}'], input['kspace'], input['mask'].float(), sample_accel=self.acceleration)


class SoftDC(nn.Module):

    def __init__(self, net, space='k-space', mode='parallel'):
        super().__init__()
        assert space in {'img-space', 'k-space'}
        assert mode in {'parallel', 'sequential'}
        self.net = net
        self.space = space
        self.mode = mode
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_buffer('zero', torch.zeros(1, 1, 1, 1, 1))

    def soft_dc(self, x, input):
        if self.space == 'img-space':
            x = ft(sens_expand(x, input['sens_maps']))
        x = torch.where(input['mask'], x - input['kspace'], self.zero)
        if self.space == 'img-space':
            x = sens_reduce(ift(x), input['sens_maps'])
        return self.lambda_ * x

    def net_forward(self, x, input):
        if self.space == 'k-space':
            x = sens_reduce(ift(x), input['sens_maps'])
        x = self.net(x)
        if self.space == 'k-space':
            x = ft(sens_expand(x, input['sens_maps']))
        return x

    def forward(self, x, input):
        if self.mode == 'parallel':
            return x - self.soft_dc(x, input) - self.net_forward(x, input)
        elif self.mode == 'sequential':
            x = self.net_forward(x, input)
            return x - self.soft_dc(x, input)


class SequentialPlus(nn.Sequential):

    def forward(self, input):
        stack = []
        x = input['kspace'].clone() if isinstance(input, dict) else input
        for module in self._modules.values():
            if isinstance(module, Push):
                stack.append(x)
            elif isinstance(module, Pop):
                if module.method == 'concat':
                    x = torch.cat((x, stack.pop()), 1)
                elif module.method == 'add':
                    x = x + stack.pop()
                else:
                    assert False
            elif isinstance(module, (DC, SensExpand, SensReduce, SoftDC, GRAPPA, MaskCenter)):
                x = module(x, input)
            else:
                x = module(x)
        return x


def pad16(x, func):

    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    b, c, h, w = x.shape
    w_mult = (w - 1 | 15) + 1
    h_mult = (h - 1 | 15) + 1
    w_pad = floor_ceil((w_mult - w) / 2)
    h_pad = floor_ceil((h_mult - h) / 2)
    x = F.pad(x, w_pad + h_pad)
    x = func(x)
    x = x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]
    return x


class Pad(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return pad16(x, self.model)


class CombineSlices(nn.Module):

    def __init__(self, slice_dim=2):
        super().__init__()
        self.slice_dim = slice_dim

    def forward(self, x):
        return torch.index_select(x, dim=self.slice_dim, index=torch.tensor(0, device=x.device))


def parse_model(s):
    if s is None:
        return None
    s = re.sub('(\\d+)\\[(.*?)\\]', lambda m: int(m.group(1)) * m.group(2), s)
    return eval(f'SequentialPlus({s})')


class SensModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_sens = parse_model(args.method_sens)
        self.model = parse_model(args.method)

    def forward(self, input):
        if self.model_sens is not None:
            input['sens_maps'] = self.model_sens(input)
        return self.model(input)


class AdversaryModel(nn.Module):
    """
        By storing the adversary in the same model object
        we avoid a lot of extra boilerplate code.
    """

    def __init__(self, prediction_model, adversary_model):
        super().__init__()
        self.prediction_model = prediction_model
        self.adversary_model = adversary_model

    def forward(self, *args, **kwargs):
        """
            Evaluating the adversary and prediction models both need
            to use the forward method to work with distributed learning
        """
        if 'adversary' in kwargs and kwargs['adversary']:
            return self.adversary_model(*args)
        else:
            return self.prediction_model(*args, **kwargs)


class AdversaryEnsemble(nn.Module):
    """
        Multiple adversaries
    """

    def __init__(self, nadvs, Adversary, **kwargs):
        super().__init__()
        self.adversaries = nn.ModuleList()
        for i in range(nadvs):
            self.adversaries.append(Adversary(**kwargs))

    def forward(self, *args):
        results = []
        for adversary in self.adversaries:
            results.append(adversary(*args))
        return torch.cat(results, dim=1)


class SSIM(nn.Module):

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = 2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2
        D = B1 * B2
        S = A1 * A2 / D
        return S.mean()


class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        win_size = 7
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, norm, max):
        q = 100000.0
        X, Y, max = X * q, Y * q, max * q
        data_range = max[:, None, None, None]
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = 2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2
        D = B1 * B2
        S = A1 * A2 / D
        return -S.mean()


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False), nn.InstanceNorm2d(out_chans), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(self, in_chans: int, out_chans: int, chans: int=32, num_pool_layers: int=4, drop_prob: float=0.0):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(nn.Sequential(ConvBlock(ch * 2, ch, drop_prob), nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1)))

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, 'reflect')
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(self, chans: int, num_pools: int, in_chans: int=2, out_chans: int=2, drop_prob: float=0.0):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.unet = Unet(in_chans=in_chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob)

    def complex_to_chan_dim(self, x: torch.Tensor) ->torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) ->torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) ->torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) ->Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = (w - 1 | 15) + 1
        h_mult = (h - 1 | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) ->torch.Tensor:
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError('Last dimension must be 2 for complex.')
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x


class AdaptiveSensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(self, chans: int, num_pools: int, in_chans: int=2, out_chans: int=2, drop_prob: float=0.0, num_sense_lines: Optional[int]=None):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
        """
        super().__init__()
        self.num_sense_lines = num_sense_lines
        self.norm_unet = NormUnet(chans, num_pools, in_chans=in_chans, out_chans=out_chans, drop_prob=drop_prob)

    def chans_to_batch_dim(self, x: torch.Tensor) ->Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) ->torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) ->torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(self, mask: torch.Tensor, num_sense_lines: Optional[int]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        if self.num_sense_lines is not None:
            if (num_low_freqs < num_sense_lines).all():
                raise RuntimeError('`num_sense_lines` cannot be greater than the actual number of low-frequency lines in the mask: {}'.format(num_low_freqs))
            num_low_freqs = num_sense_lines * torch.ones(mask.shape[0], dtype=mask.dtype, device=mask.device)
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        return pad, num_low_freqs

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, self.num_sense_lines)
        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x


class AdaptiveVarNetBlock(nn.Module):
    """
    Model block for adaptive end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, inter_sens: bool=True, hard_dc: bool=False, dc_mode: str='simul', sparse_dc_gradients: bool=True):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
            inter_sens: boolean, whether to do reduction and expansion using
                estimated sensitivity maps.
            hard_dc: boolean, whether to do hard DC layer instead of soft.
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows.
        """
        super().__init__()
        self.model = model
        self.inter_sens = inter_sens
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients
        if dc_mode not in ['first', 'last', 'simul']:
            raise ValueError("`dc_mode` must be one of 'first', 'last', or 'simul'. Not {}".format(dc_mode))
        if hard_dc:
            self.dc_weight = 1
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(self, current_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor, kspace: Optional[torch.Tensor]) ->torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1)
        if self.dc_mode == 'first':
            if self.sparse_dc_gradients:
                current_kspace = current_kspace - torch.where(mask.byte(), current_kspace - ref_kspace, zero) * self.dc_weight
            else:
                dc_kspace = current_kspace * mask
                current_kspace = current_kspace - (dc_kspace - ref_kspace) * self.dc_weight
        model_term = self.sens_expand(self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps)
        if self.dc_mode == 'first':
            return current_kspace - model_term
        elif self.dc_mode == 'simul':
            if self.sparse_dc_gradients:
                soft_dc = torch.where(mask.byte(), current_kspace - ref_kspace, zero) * self.dc_weight
            else:
                dc_kspace = current_kspace * mask
                soft_dc = (dc_kspace - ref_kspace) * self.dc_weight
            return current_kspace - soft_dc - model_term
        elif self.dc_mode == 'last':
            combined_kspace = current_kspace - model_term
            if self.sparse_dc_gradients:
                combined_kspace = combined_kspace - torch.where(mask.byte(), combined_kspace - ref_kspace, zero) * self.dc_weight
            else:
                dc_kspace = combined_kspace * mask
                combined_kspace = combined_kspace - (dc_kspace - ref_kspace) * self.dc_weight
            return combined_kspace
        else:
            raise ValueError("`dc_mode` must be one of 'first', 'last', or 'simul'. Not {}".format(self.dc_mode))


class ThresholdSigmoidMask(Function):

    def __init__(self):
        """
        Straight through estimator.
        The forward step stochastically binarizes the probability mask.
        The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdSigmoidMask, self).__init__()

    @staticmethod
    def forward(ctx, inputs, slope, clamp):
        batch_size = len(inputs)
        probs = []
        results = []
        for i in range(batch_size):
            x = inputs[i:i + 1]
            count = 0
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()
                if torch.isclose(torch.mean(result), torch.mean(x), atol=0.001):
                    break
                count += 1
                if count > 1000:
                    None
                    raise RuntimeError('Rejection sampled exceeded number of tries. Probably this means all sampling probabilities are 1 or 0 for some reason, leading to divide by zero in rescale_probs().')
            probs.append(prob)
            results.append(result)
        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        slope = torch.tensor(slope, requires_grad=False)
        ctx.clamp = clamp
        ctx.save_for_backward(inputs, probs, slope)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        input, prob, slope = ctx.saved_tensors
        if ctx.clamp:
            grad_output = F.hardtanh(grad_output)
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow(torch.exp(-slope * (input - prob)) + 1, 2)
        return current_grad * grad_output, None, None


class LOUPEPolicy(nn.Module):
    """
    LOUPE policy model.
    """

    def __init__(self, num_actions: int, budget: int, use_softplus: bool=True, slope: float=10, sampler_detach_mask: bool=False, straight_through_slope: float=10, fix_sign_leakage: bool=True, st_clamp: bool=False):
        super().__init__()
        self.use_softplus = use_softplus
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp
        if use_softplus:
            self.sampler = nn.Parameter(torch.normal(torch.ones((1, num_actions)), torch.ones((1, num_actions)) / 10))
        else:
            self.sampler = nn.Parameter(torch.zeros((1, num_actions)))
        self.binarizer = ThresholdSigmoidMask.apply
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask

    def forward(self, mask: torch.Tensor, kspace: torch.Tensor):
        B, M, H, W, C = kspace.shape
        sampler_out = self.sampler.expand(mask.shape[0], -1)
        if self.use_softplus:
            prob_mask = F.softplus(sampler_out, beta=self.slope)
            prob_mask = prob_mask / torch.max((1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])) * prob_mask, dim=1)[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * sampler_out)
        masked_prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        nonzero_idcs = (mask.view(B, W) == 0).nonzero(as_tuple=True)
        probs_to_norm = masked_prob_mask[nonzero_idcs].reshape(B, -1)
        normed_probs = self.rescale_probs(probs_to_norm)
        masked_prob_mask[nonzero_idcs] = normed_probs.flatten()
        flat_bin_mask = self.binarizer(masked_prob_mask, self.straight_through_slope, self.st_clamp)
        acquisitions = flat_bin_mask.reshape(B, 1, 1, W, 1)
        final_prob_mask = masked_prob_mask.reshape(B, 1, 1, W, 1)
        mask = mask + acquisitions
        masked_kspace = mask * kspace
        if self.sampler_detach_mask:
            mask = mask.detach()
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, final_prob_mask

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """
        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i:i + 1]
            xbar = torch.mean(x)
            r = sparsity / xbar
            beta = (1 - sparsity) / (1 - xbar)
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))
        return torch.cat(ret, dim=0)


class SingleConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float=0, pool_size: int=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
            pool_size (int): Size of 2D max-pooling operator.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.pool_size = pool_size
        layers = [nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1), nn.InstanceNorm2d(out_chans), nn.ReLU(), nn.Dropout2d(drop_prob)]
        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, drop_prob={self.drop_prob}, max_pool_size={self.pool_size})'


class LineConvSampler(nn.Module):

    def __init__(self, input_dim: tuple=(2, 128, 128), chans: int=16, num_pool_layers: int=4, fc_size: int=256, drop_prob: float=0, slope: float=10, use_softplus: bool=True, num_fc_layers: int=3, activation: str='leakyrelu'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            input_dim (tuple): Input size of reconstructed images (C, H, W).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling layers.
            fc_size (int): Number of hidden neurons for the fully connected layers.
            drop_prob (float): Dropout probability.
            num_fc_layers (int): Number of fully connected layers to use after convolutional part.
            use_softplus (bool): Whether to use softplus as final activation (otherwise sigmoid).
            activation (str): Activation function to use: leakyrelu or elu.
        """
        super().__init__()
        assert len(input_dim) == 3
        self.input_dim = input_dim
        self.in_chans = input_dim[0]
        self.num_actions = input_dim[-1]
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.pool_size = 2
        self.slope = slope
        self.use_softplus = use_softplus
        self.num_fc_layers = num_fc_layers
        self.activation = activation
        self.channel_layer = SingleConvBlock(self.in_chans, chans, drop_prob, pool_size=1)
        self.down_sample_layers = nn.ModuleList([SingleConvBlock(chans * 2 ** i, chans * 2 ** (i + 1), drop_prob, pool_size=self.pool_size) for i in range(num_pool_layers)])
        self.feature_extractor = nn.Sequential(self.channel_layer, *self.down_sample_layers)
        self.flattened_size = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
        fc_out: List[nn.Module] = []
        for layer in range(self.num_fc_layers):
            in_features = fc_size
            out_features = fc_size
            if layer == 0:
                in_features = self.flattened_size
            if layer + 1 == self.num_fc_layers:
                out_features = self.num_actions
            fc_out.append(nn.Linear(in_features=in_features, out_features=out_features))
            if layer + 1 < self.num_fc_layers:
                act: nn.Module
                if activation == 'leakyrelu':
                    act = nn.LeakyReLU()
                elif activation == 'elu':
                    act = nn.ELU()
                else:
                    raise RuntimeError(f'Invalid activation function {activation}. Should be leakyrelu or elu.')
                fc_out.append(act)
        self.fc_out = nn.Sequential(*fc_out)

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
            mask (torch.Tensor): Input tensor of shape [resolution], containing 0s and 1s
        Returns:
            torch.Tensor: prob_mask [batch_size, num_actions] corresponding to all actions at the
            given observation. Gives probabilities of sampling a particular action.
        """
        image_emb = self.feature_extractor(image)
        image_emb = image_emb.flatten(start_dim=1)
        out = self.fc_out(image_emb)
        if self.use_softplus:
            out = F.softplus(out, beta=self.slope)
            prob_mask = out / torch.max((1 - mask.reshape(out.shape[0], out.shape[1])) * out, dim=1)[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * out)
        prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        assert len(prob_mask.shape) == 2
        return prob_mask


class StraightThroughPolicy(nn.Module):
    """
    Policy model for active acquisition.
    """

    def __init__(self, budget: int, crop_size: Tuple[int, int]=(128, 128), slope: float=10, sampler_detach_mask: bool=False, use_softplus: bool=True, straight_through_slope: float=10, fix_sign_leakage: bool=True, st_clamp: bool=False, fc_size: int=256, drop_prob: float=0.0, num_fc_layers: int=3, activation: str='leakyrelu'):
        super().__init__()
        self.sampler = LineConvSampler(input_dim=(2, *crop_size), slope=slope, use_softplus=use_softplus, fc_size=fc_size, num_fc_layers=num_fc_layers, drop_prob=drop_prob, activation=activation)
        self.binarizer = ThresholdSigmoidMask.apply
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask
        self.use_softplus = use_softplus
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.num_fc_layers = num_fc_layers
        self.activation = activation

    def forward(self, kspace_pred: torch.Tensor, mask: torch.Tensor):
        B, C, H, W = kspace_pred.shape
        flat_prob_mask = self.sampler(kspace_pred, mask)
        nonzero_idcs = (mask.view(B, W) == 0).nonzero(as_tuple=True)
        probs_to_norm = flat_prob_mask[nonzero_idcs].reshape(B, -1)
        normed_probs = self.rescale_probs(probs_to_norm)
        flat_prob_mask[nonzero_idcs] = normed_probs.flatten()
        flat_bin_mask = self.binarizer(flat_prob_mask, self.straight_through_slope, self.st_clamp)
        return flat_bin_mask, flat_prob_mask

    def do_acquisition(self, kspace: torch.Tensor, kspace_pred: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor):
        B, M, H, W, C = kspace.shape
        current_recon = self.sens_reduce(kspace_pred, sens_maps).squeeze(1).permute(0, 3, 1, 2)
        acquisitions, flat_prob_mask = self(current_recon, mask)
        acquisitions = acquisitions.reshape(B, 1, 1, W, 1)
        prob_mask = flat_prob_mask.reshape(B, 1, 1, W, 1)
        mask = mask + acquisitions
        masked_kspace = mask * kspace
        if self.sampler_detach_mask:
            mask = mask.detach()
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, prob_mask

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """
        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i:i + 1]
            xbar = torch.mean(x)
            r = sparsity / xbar
            beta = (1 - sparsity) / (1 - xbar)
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))
        return torch.cat(ret, dim=0)


class AdaptiveVarNet(nn.Module):
    """
    A full adaptive variational network model. This model uses a policy to do
    end-to-end adaptive acquisition and reconstruction.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(self, budget: int=22, num_cascades: int=12, sens_chans: int=8, sens_pools: int=4, chans: int=18, pools: int=4, cascades_per_policy: int=1, loupe_mask: bool=False, use_softplus: bool=True, crop_size: Tuple[int, int]=(128, 128), num_actions: Optional[int]=None, num_sense_lines: Optional[int]=None, hard_dc: bool=False, dc_mode: str='simul', slope: float=10, sparse_dc_gradients: bool=True, straight_through_slope: float=10, st_clamp: bool=False, policy_fc_size: int=256, policy_drop_prob: float=0.0, policy_num_fc_layers: int=3, policy_activation: str='leakyrelu'):
        """
        Args:
            budget: Total number of acquisitions to do.
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            cascades_per_policy: How many cascades to use per policy step.
                Policies will be applied starting after first cascade, and then
                every cascades_per_policy cascades after. Note that
                num_cascades % cascades_per_policy should equal 1. There is an
                option to set cascades_per_policy equal to num_cascades as well,
                in which case the policy will be applied before the first
                cascade only.
            loupe_mask: Whether to use LOUPE-like mask instead of equispaced
                (still keeps center lines).
            use_softplus: Whether to use softplus or sigmoid in LOUPE.
            crop_size: tuple, crop size of MR images.
            num_actions: Number of possible actions to sample (=image width).
                Used only when loupe_mask is True.
            num_sense_lines: Number of low-frequency lines to use for
                sensitivity map computation, must be even or `None`. Default
                `None` will automatically compute the number from masks.
                Default behaviour may cause some slices to use more
                low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: Whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            slope: Slope to use for sigmoid in LOUPE and Policy forward, or
                beta to use in softplus.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows.
            straight_through_slope: Slope to use in Straight Through estimator.
            st_clamp: Whether to clamp gradients between -1 and 1 in straight
                through estimator.
            policy_fc_size: int, size of fully connected layers in Policy
                architecture.
            policy_drop_prob: float, dropout probability of convolutional
                layers in Policy.
            policy_num_fc_layers: int, number of fully-connected layers to
                apply after the convolutional layers in the policy.
            policy_activation: str, "leakyrelu" or "elu". Activation function
                to use between fully-connected layers in the policy. Only used
                if policy_num_fc_layers > 1.
        """
        super().__init__()
        self.budget = budget
        self.cascades_per_policy = cascades_per_policy
        self.loupe_mask = loupe_mask
        self.use_softplus = use_softplus
        self.crop_size = crop_size
        self.num_actions = num_actions
        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.slope = slope
        self.sparse_dc_gradients = sparse_dc_gradients
        self.straight_through_slope = straight_through_slope
        self.st_clamp = st_clamp
        self.policy_fc_size = policy_fc_size
        self.policy_drop_prob = policy_drop_prob
        self.policy_num_fc_layers = policy_num_fc_layers
        self.policy_activation = policy_activation
        self.sens_net = AdaptiveSensitivityModel(sens_chans, sens_pools, num_sense_lines=num_sense_lines)
        self.cascades = nn.ModuleList([AdaptiveVarNetBlock(NormUnet(chans, pools), hard_dc=hard_dc, dc_mode=dc_mode, sparse_dc_gradients=sparse_dc_gradients) for _ in range(num_cascades)])
        if self.loupe_mask:
            assert isinstance(self.num_actions, int)
            self.loupe = LOUPEPolicy(self.num_actions, self.budget, use_softplus=self.use_softplus, slope=self.slope, straight_through_slope=self.straight_through_slope, st_clamp=self.st_clamp)
        else:
            remaining_budget = self.budget
            if cascades_per_policy > num_cascades:
                raise RuntimeError('Number of cascades {} cannot be smaller than number of cascades per policy {}.'.format(num_cascades, cascades_per_policy))
            elif num_cascades != cascades_per_policy:
                base_budget = self.budget // ((num_cascades - 1) // cascades_per_policy)
                policies = []
                for i in range(1, num_cascades):
                    if (num_cascades - i) % cascades_per_policy == 0:
                        if remaining_budget < 2 * base_budget:
                            policy = StraightThroughPolicy(remaining_budget, crop_size, slope=self.slope, use_softplus=self.use_softplus, straight_through_slope=self.straight_through_slope, st_clamp=self.st_clamp, fc_size=self.policy_fc_size, drop_prob=self.policy_drop_prob, num_fc_layers=self.policy_num_fc_layers, activation=self.policy_activation)
                            remaining_budget = 0
                        else:
                            policy = StraightThroughPolicy(base_budget, crop_size, slope=self.slope, use_softplus=self.use_softplus, straight_through_slope=self.straight_through_slope, st_clamp=self.st_clamp, fc_size=self.policy_fc_size, drop_prob=self.policy_drop_prob, num_fc_layers=self.policy_num_fc_layers, activation=self.policy_activation)
                            remaining_budget -= base_budget
                        policies.append(policy)
            else:
                policies = [StraightThroughPolicy(self.budget, crop_size, slope=self.slope, use_softplus=self.use_softplus, straight_through_slope=self.straight_through_slope, st_clamp=self.st_clamp, fc_size=self.policy_fc_size, drop_prob=self.policy_drop_prob, num_fc_layers=self.policy_num_fc_layers, activation=self.policy_activation)]
            self.policies = nn.ModuleList(policies)

    def forward(self, kspace: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor):
        extra_outputs = defaultdict(list)
        mask, masked_kspace = self.extract_low_freq_mask(mask, masked_kspace)
        extra_outputs['masks'].append(mask)
        sens_maps = self.sens_net(masked_kspace, mask)
        extra_outputs['sense'].append(sens_maps)
        current_recon = fastmri.complex_abs(self.sens_reduce(masked_kspace, sens_maps)).squeeze(1)
        extra_outputs['recons'].append(current_recon.detach().cpu())
        if self.loupe_mask:
            mask, masked_kspace, prob_mask = self.loupe(mask, kspace)
            extra_outputs['masks'].append(mask)
            extra_outputs['prob_masks'].append(prob_mask)
            current_recon = fastmri.complex_abs(self.sens_reduce(masked_kspace, sens_maps)).squeeze(1)
            extra_outputs['recons'].append(current_recon.detach().cpu())
        if self.cascades_per_policy == len(self.cascades) and not self.loupe_mask:
            if len(self.policies) != 1:
                raise ValueError(f'Must have only one policy when number of cascades {len(self.cascades)} equals the number of cascades_per_policy {self.cascades_per_policy}.')
            kspace_pred = masked_kspace.clone()
            mask, masked_kspace, prob_mask = self.policies[0].do_acquisition(kspace, kspace_pred, mask, sens_maps)
            extra_outputs['masks'].append(mask)
            extra_outputs['prob_masks'].append(prob_mask)
        kspace_pred = masked_kspace.clone()
        j = 0
        for i, cascade in enumerate(self.cascades):
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, kspace=kspace)
            current_recon = fastmri.complex_abs(self.sens_reduce(masked_kspace, sens_maps)).squeeze(1)
            extra_outputs['recons'].append(current_recon.detach().cpu())
            if i == len(self.cascades) - 1 or self.loupe_mask:
                continue
            if (len(self.cascades) - (i + 1)) % self.cascades_per_policy == 0 and self.cascades_per_policy != len(self.cascades):
                mask, masked_kspace, prob_mask = self.policies[j].do_acquisition(kspace, kspace_pred, mask, sens_maps)
                j += 1
                extra_outputs['masks'].append(mask)
                extra_outputs['prob_masks'].append(prob_mask)
        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        extra_outputs['recons'].append(output.detach().cpu())
        return output, extra_outputs

    def extract_low_freq_mask(self, mask: torch.Tensor, masked_kspace: torch.Tensor):
        """
        Extracts low frequency components that are used by sensitivity map
        computation. This serves as the starting point for active acquisition.
        """
        pad, num_low_freqs = self.sens_net.get_pad_and_num_low_freqs(mask, self.num_sense_lines)
        mask = transforms.batched_mask_center(mask, pad, pad + num_low_freqs)
        masked_kspace = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        return mask, masked_kspace

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(self, chans: int, num_pools: int, in_chans: int=2, out_chans: int=2, drop_prob: float=0.0, mask_center: bool=True):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(chans, num_pools, in_chans=in_chans, out_chans=out_chans, drop_prob=drop_prob)

    def chans_to_batch_dim(self, x: torch.Tensor) ->Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) ->torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) ->torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(self, mask: torch.Tensor, num_low_frequencies: Optional[int]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0]
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(mask.shape[0], dtype=mask.dtype, device=mask.device)
        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad, num_low_frequencies_tensor

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, num_low_frequencies: Optional[int]=None) ->torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))
        return self.divide_root_sum_of_squares(self.batch_chans_to_chan_dim(self.norm_unet(images), batches))


class VarNet(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(self, num_cascades: int=12, sens_chans: int=8, sens_pools: int=4, chans: int=18, pools: int=4, num_sense_lines: Optional[int]=None, hard_dc: bool=False, dc_mode: str='simul', sparse_dc_gradients: bool=True):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            num_sense_lines: Number of low-frequency lines to use for
                sensitivity map computation, must be even or `None`. Default
                `None` will automatically compute the number from masks.
                Default behaviour may cause some slices to use more
                low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows. This should change
                nothing for the non-active VarNet.
        """
        super().__init__()
        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients
        self.sens_net = AdaptiveSensitivityModel(sens_chans, sens_pools, num_sense_lines=num_sense_lines)
        self.cascades = nn.ModuleList([AdaptiveVarNetBlock(NormUnet(chans, pools), hard_dc=hard_dc, dc_mode=dc_mode, sparse_dc_gradients=sparse_dc_gradients) for _ in range(num_cascades)])

    def forward(self, kspace: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor):
        extra_outputs = defaultdict(list)
        sens_maps = self.sens_net(masked_kspace, mask)
        extra_outputs['sense'].append(sens_maps.detach().cpu())
        kspace_pred = masked_kspace.clone()
        extra_outputs['masks'].append(mask.detach().cpu())
        current_recon = fastmri.complex_abs(self.sens_reduce(masked_kspace, sens_maps)).squeeze(1)
        extra_outputs['recons'].append(current_recon.detach().cpu())
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, kspace=kspace)
            current_recon = fastmri.complex_abs(self.sens_reduce(masked_kspace, sens_maps)).squeeze(1)
            extra_outputs['recons'].append(current_recon.detach().cpu())
        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        return output, extra_outputs

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        return fastmri.complex_mul(fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(self, current_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) ->torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps)
        return current_kspace - soft_dc - model_term


class NMSELoss(nn.Module):

    def forward(self, X, Y, norm, max):
        se = torch.sum((X - Y) ** 2, dim=(1, 2, 3))
        return torch.sum(se / norm ** 2)


class L1Loss(nn.Module):

    def forward(self, X, Y, norm, max):
        return F.l1_loss(X, Y)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdversaryModel,
     lambda: ([], {'prediction_model': _mock_layer(), 'adversary_model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cartesian,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CombineSlices,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_chans': 4, 'out_chans': 4, 'drop_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Fm2Batch,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fm2Complex,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResnetBlock,
     lambda: ([], {'fin': 256, 'fout': 32}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 1])], {}),
     True),
    (SSIMLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 1]), torch.rand([4, 1, 1])], {}),
     True),
    (SequentialPlus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SingleConvBlock,
     lambda: ([], {'in_chans': 4, 'out_chans': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransposeConvBlock,
     lambda: ([], {'in_chans': 4, 'out_chans': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnpooledResnet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (WideDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_facebookresearch_fastMRI(_paritybench_base):
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

