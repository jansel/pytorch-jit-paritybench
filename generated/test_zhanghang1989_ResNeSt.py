import sys
_module = sys.modules[__name__]
del sys
prepare_coco = _module
train_net = _module
hubconf = _module
resnest = _module
d2 = _module
config = _module
resnest = _module
splat = _module
gluon = _module
ablation = _module
data_utils = _module
dropblock = _module
model_store = _module
model_zoo = _module
resnet = _module
transforms = _module
datasets = _module
build = _module
imagenet = _module
loss = _module
models = _module
ablation = _module
resnest = _module
resnet = _module
splat = _module
autoaug = _module
build = _module
utils = _module
prepare_imagenet = _module
train = _module
verify = _module
train = _module
verify = _module
setup = _module
test_gluon = _module
test_radix_major = _module
test_torch = _module

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


from collections import OrderedDict


import torch


import numpy as np


import torch.nn.functional as F


from torch import nn


from torch.nn import Module


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


import torch.nn as nn


from torch.autograd import Variable


import math


from torch.nn import Conv2d


from torchvision.transforms import *


import functools


import time


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import warnings


import inspect


class ResNetBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BasicBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, stride=1, norm='BN'):
        """
        The standard block type for ResNet18 and ResNet34.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): A callable that takes the number of
                channels and returns a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1, avd=False, avg_down=False, radix=2, bottleneck_width=64):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.avd = avd and stride > 1
        self.avg_down = avg_down
        self.radix = radix
        cardinality = num_groups
        group_width = int(bottleneck_channels * (bottleneck_width / 64.0)) * cardinality
        if in_channels != out_channels:
            if self.avg_down:
                self.shortcut_avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, norm=get_norm(norm, out_channels))
            else:
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, group_width, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, group_width))
        if self.radix > 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=dilation, dilation=dilation, groups=cardinality, bias=False, radix=self.radix, norm=norm)
        else:
            self.conv2 = Conv2d(group_width, group_width, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, group_width))
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        self.conv3 = Conv2d(group_width, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        if self.radix > 1:
            for layer in [self.conv1, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        if self.radix > 1:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = F.relu_(out)
        if self.avd:
            out = self.avd_layer(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            if self.avg_down:
                x = self.shortcut_avgpool(x)
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1, deform_modulated=False, deform_num_groups=1, avd=False, avg_down=False, radix=2, bottleneck_width=64):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated
        self.avd = avd and stride > 1
        self.avg_down = avg_down
        self.radix = radix
        cardinality = num_groups
        group_width = int(bottleneck_channels * (bottleneck_width / 64.0)) * cardinality
        if in_channels != out_channels:
            if self.avg_down:
                self.shortcut_avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, norm=get_norm(norm, out_channels))
            else:
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, group_width, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, group_width))
        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18
        self.conv2_offset = Conv2d(bottleneck_channels, offset_channels * deform_num_groups, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=1 * dilation, dilation=dilation, groups=deform_num_groups)
        if self.radix > 1:
            self.conv2 = SplAtConv2d_dcn(group_width, group_width, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=dilation, dilation=dilation, groups=cardinality, bias=False, radix=self.radix, norm=norm, deform_conv_op=deform_conv_op, deformable_groups=deform_num_groups, deform_modulated=deform_modulated)
        else:
            self.conv2 = deform_conv_op(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, deformable_groups=deform_num_groups, norm=get_norm(norm, bottleneck_channels))
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        self.conv3 = Conv2d(group_width, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        if self.radix > 1:
            for layer in [self.conv1, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        if self.radix > 1:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            if self.deform_modulated:
                offset_mask = self.conv2_offset(out)
                offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat((offset_x, offset_y), dim=1)
                mask = mask.sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = F.relu_(out)
        if self.avd:
            out = self.avd_layer(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            if self.avg_down:
                x = self.shortcut_avgpool(x)
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, norm='BN', deep_stem=False, stem_width=32):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.deep_stem = deep_stem
        if self.deep_stem:
            self.conv1_1 = Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, norm=get_norm(norm, stem_width))
            self.conv1_2 = Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, stem_width))
            self.conv1_3 = Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, stem_width * 2))
            for layer in [self.conv1_1, self.conv1_2, self.conv1_3]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
            weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        if self.deep_stem:
            x = self.conv1_1(x)
            x = F.relu_(x)
            x = self.conv1_2(x)
            x = F.relu_(x)
            x = self.conv1_3(x)
            x = F.relu_(x)
        else:
            x = self.conv1(x)
            x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        if self.deep_stem:
            return self.conv1_3.out_channels
        else:
            return self.conv1.out_channels

    @property
    def stride(self):
        return 4


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class SplAtConv2d_dcn(Module):
    """Split-Attention Conv2d with dcn
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm=None, dropblock_prob=0.0, deform_conv_op=None, deformable_groups=1, deform_modulated=False, **kwargs):
        super(SplAtConv2d_dcn, self).__init__()
        self.deform_modulated = deform_modulated
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = deform_conv_op(in_channels, channels * radix, kernel_size, stride, padding[0], dilation, groups=groups * radix, bias=bias, deformable_groups=deformable_groups, **kwargs)
        self.use_bn = norm is not None
        if self.use_bn:
            self.bn0 = get_norm(norm, channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x, offset_input):
        if self.deform_modulated:
            offset_x, offset_y, mask = torch.chunk(offset_input, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            x = self.conv(x, offset, mask)
        else:
            x = self.conv(x, offset_input)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class NLLMultiLabelSmooth(nn.Module):

    def __init__(self, smoothing=0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False, rectified_conv=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)
        if radix >= 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=cardinality, bias=False, radix=radix, rectify=rectified_conv, rectify_avg=rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        elif rectified_conv:
            self.conv2 = RFConv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=cardinality, bias=False, average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64, num_classes=1000, dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, rectified_conv=False, rectify_avg=False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0, last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs), norm_layer(stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs), norm_layer(stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=dilation, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
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
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        return x


class RadixMajorNaiveImp(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(RadixMajorNaiveImp, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        assert not self.use_bn
        self.relu = ReLU(inplace=True)
        cardinal_group_width = channels // groups
        cardinal_inter_channels = inter_channels // groups
        self.fc1 = nn.ModuleList([nn.Linear(cardinal_group_width, cardinal_inter_channels) for _ in range(groups)])
        self.fc2 = nn.ModuleList([nn.Linear(cardinal_inter_channels, cardinal_group_width * radix) for _ in range(groups)])
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, channel = x.shape[:2]
        cardinality = self.cardinality
        radix = self.radix
        tiny_group_width = channel // radix // cardinality
        all_groups = torch.split(x, tiny_group_width, dim=1)
        out = []
        for k in range(cardinality):
            U_k = [all_groups[r * cardinality + k] for r in range(radix)]
            U_k = sum(U_k)
            gap_k = F.adaptive_avg_pool2d(U_k, 1).squeeze()
            atten_k = self.fc2[k](self.fc1[k](gap_k))
            if radix > 1:
                x_k = [all_groups[r * cardinality + k] for r in range(radix)]
                x_k = torch.cat(x_k, dim=1)
                atten_k = atten_k.view(batch, radix, -1)
                atten_k = F.softmax(atten_k, dim=1)
            else:
                x_k = all_groups[k]
                atten_k = torch.sigmoid(atten_k)
            attended_k = x_k * atten_k.view(batch, -1, 1, 1)
            out_k = sum(torch.split(attended_k, attended_k.size(1) // self.radix, dim=1))
            out.append(out_k)
        return torch.cat(out, dim=1).contiguous()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLLMultiLabelSmooth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RadixMajorNaiveImp,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SplAtConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (rSoftMax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zhanghang1989_ResNeSt(_paritybench_base):
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

