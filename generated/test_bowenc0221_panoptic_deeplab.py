import sys
_module = sys.modules[__name__]
del sys
prepare_coco_panoptic_trainid = _module
segmentation = _module
config = _module
default = _module
hrnet_config = _module
data = _module
build = _module
datasets = _module
base_dataset = _module
cityscapes = _module
cityscapes_panoptic = _module
coco_panoptic = _module
utils = _module
samplers = _module
distributed_sampler = _module
transforms = _module
pre_augmentation_transforms = _module
target_transforms = _module
transforms = _module
evaluation = _module
coco_instance = _module
instance = _module
panoptic = _module
semantic = _module
model = _module
backbone = _module
hrnet = _module
mnasnet = _module
mobilenet = _module
resnet = _module
xception = _module
build = _module
decoder = _module
aspp = _module
conv_module = _module
deeplabv3 = _module
deeplabv3plus = _module
panoptic_deeplab = _module
loss = _module
criterion = _module
meta_arch = _module
base = _module
deeplabv3 = _module
deeplabv3plus = _module
panoptic_deeplab = _module
post_processing = _module
evaluation_format = _module
instance_post_processing = _module
semantic_post_processing = _module
solver = _module
build = _module
lr_scheduler = _module
comm = _module
debug = _module
env = _module
flow_vis = _module
logger = _module
save_annotation = _module
test_utils = _module
utils = _module
_init_paths = _module
demo = _module
test_net_single_core = _module
train_net = _module
d2 = _module
backbone = _module
predictor = _module
train_deeplab = _module
train_panoptic_deeplab = _module

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


import torch


import numpy as np


from torch.utils import data


import itertools


import math


from collections import defaultdict


from typing import Optional


from torch.utils.data.sampler import Sampler


import random


from torchvision.transforms import functional as F


import torch.nn as nn


import torch.nn.functional as F


import warnings


from torch import nn


from collections import OrderedDict


from torch.nn import functional as F


from functools import partial


from enum import Enum


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Set


from typing import Type


from typing import Union


import functools


import torch.distributed as dist


import time


import torch.backends.cudnn as cudnn


from torch.nn.parallel import DistributedDataParallel


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


logger = logging.getLogger('hrnet_backbone')


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(num_channels[branch_index] * block.expansion))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), self.norm_layer(num_outchannels_conv3x3), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, norm_layer=None):
        super(HighResolutionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), self.norm_layer(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), self.norm_layer(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output, norm_layer=self.norm_layer))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        outputs = {}
        outputs['res2'] = x[0]
        outputs['res3'] = x[1]
        outputs['res4'] = x[2]
        outputs['res5'] = x[3]
        return outputs


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


_BN_MOMENTUM = 1 - 0.9997


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM), _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM), _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM), _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM), _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM), _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM), _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM)]
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        outputs = {}
        for i, module in enumerate(self.layers):
            x = module(x)
            outputs['layer_%d' % (i + 1)] = x
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.zeros_(m.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        assert version in [1, 2]
        if version == 1 and not self.alpha == 1.0:
            depths = _get_depths(self.alpha)
            v1_stem = [nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(16, momentum=_BN_MOMENTUM), _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM)]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer
            self._version = 1
            warnings.warn('A new version of MNASNet model has been implemented. Your checkpoint was saved using the previous version. This checkpoint will load and work as before, but you may want to upgrade by training a newer model or transfer learning from an updated ImageNet checkpoint.', UserWarning)
        super(MNASNet, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None):
        """
        MobileNet V2 main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        outputs = {}
        for i, module in enumerate(self.features):
            x = module(x)
            outputs['layer_%d' % (i + 1)] = x
        return outputs

    def forward(self, x):
        return self._forward_impl(x)


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
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

    def _forward_impl(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outputs['stem'] = x
        x = self.layer1(x)
        outputs['res2'] = x
        x = self.layer2(x)
        outputs['res3'] = x
        x = self.layer3(x)
        outputs['res4'] = x
        x = self.layer4(x)
        outputs['res5'] = x
        return outputs

    def forward(self, x):
        return self._forward_impl(x)


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)
        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()), ('depthwise', depthwise), ('bn_depth', bn_depth), ('pointwise', pointwise), ('bn_point', bn_point)]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise), ('bn_depth', bn_depth), ('relu1', nn.ReLU(inplace=True)), ('pointwise', pointwise), ('bn_point', bn_point), ('relu2', nn.ReLU(inplace=True))]))

    def forward(self, x):
        return self.block(x)


class XceptionBlock(nn.Module):

    def __init__(self, channel_list, stride=1, dilation=1, skip_connection_type='conv', relu_first=True, low_feat=False, norm_layer=nn.BatchNorm2d):
        super(XceptionBlock, self).__init__()
        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat
        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1, stride=stride, bias=False)
            self.bn = norm_layer(channel_list[-1])
        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1], dilation=dilation, relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2], dilation=dilation, relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3], dilation=dilation, relu_first=relu_first, stride=stride, norm_layer=norm_layer)
        self.last_inp_channels = channel_list[3]

    def forward(self, inputs):
        sc1 = self.sep_conv1(inputs)
        sc2 = self.sep_conv2(sc1)
        residual = self.sep_conv3(sc2)
        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.bn(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')
        if self.low_feat:
            return outputs, sc2
        else:
            return outputs


class Xception65(nn.Module):

    def __init__(self, replace_stride_with_dilation=None, norm_layer=None):
        super(Xception65, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        if replace_stride_with_dilation[1]:
            assert replace_stride_with_dilation[2]
            output_stride = 8
        elif replace_stride_with_dilation[2]:
            output_stride = 16
        else:
            output_stride = 32
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
            exit_block_stride = 2
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
            exit_block_stride = 1
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
            exit_block_stride = 1
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2, norm_layer=norm_layer)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2, low_feat=True, norm_layer=norm_layer)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=entry_block3_stride, low_feat=True, norm_layer=norm_layer)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum', norm_layer=norm_layer)
        self.block20 = XceptionBlock([728, 728, 1024, 1024], stride=exit_block_stride, dilation=exit_block_dilations[0], norm_layer=norm_layer)
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=exit_block_dilations[1], skip_connection_type='none', relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        outputs['stem'] = x
        x = self.block1(x)
        x, c1 = self.block2(x)
        outputs['res2'] = c1
        x, c2 = self.block3(x)
        outputs['res3'] = c2
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        c3 = self.block19(x)
        outputs['res4'] = c3
        x = self.block20(c3)
        c4 = self.block21(x)
        outputs['res5'] = c4
        return outputs


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.ReLU())

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = nn.AdaptiveAvgPool2d(1)
        else:
            self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Decoder(nn.Module):

    def __init__(self, in_channels, feature_key, decoder_channels, atrous_rates, num_classes):
        super(DeepLabV3Decoder, self).__init__()
        self.aspp = ASPP(in_channels, out_channels=decoder_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.classifier = nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False), nn.BatchNorm2d(decoder_channels), nn.ReLU(), nn.Conv2d(decoder_channels, num_classes, 1))

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()
        res5 = features[self.feature_key]
        x = self.aspp(res5)
        x = self.classifier(x)
        pred['semantic'] = x
        return pred


def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1, with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=has_bias))
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1, with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups
    module = []
    module.extend([basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, with_bn=True, with_relu=True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)])
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def stacked_conv(in_planes, out_planes, kernel_size, num_stack, stride=1, padding=1, groups=1, with_bn=True, with_relu=True, conv_type='basic_conv'):
    """stacked convolution with bn and relu"""
    if num_stack < 1:
        assert ValueError('`num_stack` has to be a positive integer.')
    if conv_type == 'basic_conv':
        conv = partial(basic_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, with_bn=with_bn, with_relu=with_relu)
    elif conv_type == 'depthwise_separable_conv':
        conv = partial(depthwise_separable_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, with_bn=with_bn, with_relu=with_relu)
    else:
        raise ValueError('Unknown conv_type: {}'.format(conv_type))
    module = []
    module.append(conv(in_planes=in_planes))
    for n in range(1, num_stack):
        module.append(conv(in_planes=out_planes))
    return nn.Sequential(*module)


class DeepLabV3PlusDecoder(nn.Module):

    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.aspp = ASPP(in_channels, out_channels=decoder_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.low_level_key = low_level_key
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, low_level_channels_project, 1, bias=False), nn.BatchNorm2d(low_level_channels_project), nn.ReLU())
        self.fuse = stacked_conv(decoder_channels + low_level_channels_project, decoder_channels, kernel_size=3, padding=1, num_stack=2, conv_type='depthwise_separable_conv')
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()
        l = features[self.low_level_key]
        x = features[self.feature_key]
        x = self.aspp(x)
        l = self.project(l)
        x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, l), dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        pred['semantic'] = x
        return pred


class SinglePanopticDeepLabDecoder(nn.Module):

    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, aspp_channels=None):
        super(SinglePanopticDeepLabDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        project = []
        fuse = []
        for i in range(self.decoder_stage):
            project.append(nn.Sequential(nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False), nn.BatchNorm2d(low_level_channels_project[i]), nn.ReLU()))
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(fuse_conv(fuse_in_channels, decoder_channels))
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)
        return x


class SinglePanopticDeepLabHead(nn.Module):

    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)
        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(fuse_conv(decoder_channels, head_channels), nn.Conv2d(head_channels, num_classes[i], 1))
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        for key in self.class_key:
            pred[key] = self.classifier[key](x)
        return pred


class PanopticDeepLabDecoder(nn.Module):

    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes, **kwargs):
        super(PanopticDeepLabDecoder, self).__init__()
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])
        self.instance_decoder = None
        self.instance_head = None
        if kwargs.get('has_instance', False):
            instance_decoder_kwargs = dict(in_channels=in_channels, feature_key=feature_key, low_level_channels=low_level_channels, low_level_key=low_level_key, low_level_channels_project=kwargs['instance_low_level_channels_project'], decoder_channels=kwargs['instance_decoder_channels'], atrous_rates=atrous_rates, aspp_channels=kwargs['instance_aspp_channels'])
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(decoder_channels=kwargs['instance_decoder_channels'], head_channels=kwargs['instance_head_channels'], num_classes=kwargs['instance_num_classes'], class_key=kwargs['instance_class_key'])
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
        return pred


class RegularCE(nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def forward(self, logits, labels, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label
        pixel_losses = pixel_losses[mask]
        return pixel_losses.mean()


class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, threshold=0.7, min_kept=100000, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def forward(self, logits, labels, **kwargs):
        predictions = F.softmax(logits, dim=1)
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label
        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = predictions.contiguous().view(-1)[mask].contiguous().sort()
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)
        pixel_losses = pixel_losses[mask][indices]
        pixel_losses = pixel_losses[predictions < threshold]
        return pixel_losses.mean()


class DeepLabCE(nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def forward(self, logits, labels, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()
        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class BaseSegmentationModel(nn.Module):
    """
    Base class for segmentation models.
    Arguments:
        backbone: A nn.Module of backbone model.
        decoder: A nn.Module of decoder.
    """

    def __init__(self, backbone, decoder):
        super(BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder

    def _init_params(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_image_pooling(self, pool_size):
        self.decoder.set_image_pooling(pool_size)

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            result[key] = out
        return result

    def forward(self, x, targets=None):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        pred = self.decoder(features)
        results = self._upsample_predictions(pred, input_shape)
        if targets is None:
            return results
        else:
            return self.loss(results, targets)

    def loss(self, results, targets=None):
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class DeepLabV3(BaseSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        in_channels (int): number of input channels from the backbone
        feature_key (str): name of input feature from backbone
        decoder_channels (int): number of channels in decoder
        atrous_rates (tuple): atrous rates for ASPP
        num_classes (int): number of classes
        semantic_loss (nn.Module): loss function
        semantic_loss_weight (float): loss weight
    """

    def __init__(self, backbone, in_channels, feature_key, decoder_channels, atrous_rates, num_classes, semantic_loss, semantic_loss_weight, **kwargs):
        decoder = DeepLabV3Decoder(in_channels, feature_key, decoder_channels, atrous_rates, num_classes)
        super(DeepLabV3, self).__init__(backbone, decoder)
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()
        self._init_params()

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        if targets is not None:
            semantic_loss = self.semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            results['loss'] = semantic_loss
        return results


class DeepLabV3Plus(BaseSegmentationModel):
    """
    Implements DeepLabV3+ model from
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        in_channels (int): number of input channels from the backbone
        feature_key (str): name of input feature from backbone
        low_level_channels (int): channels of low-level features
        low_level_key (str): name of low-level features used in decoder
        low_level_channels_project (int): channels of low-level features after projection in decoder
        decoder_channels (int): number of channels in decoder
        atrous_rates (tuple): atrous rates for ASPP
        num_classes (int): number of classes
        semantic_loss (nn.Module): loss function
        semantic_loss_weight (float): loss weight
    """

    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes, semantic_loss, semantic_loss_weight, **kwargs):
        decoder = DeepLabV3PlusDecoder(in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes)
        super(DeepLabV3Plus, self).__init__(backbone, decoder)
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()
        self._init_params()

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        if targets is not None:
            semantic_loss = self.semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            results['loss'] = semantic_loss
        return results


class PanopticDeepLab(BaseSegmentationModel):
    """
    Implements Panoptic-DeepLab model from
    `"Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation"
    <https://arxiv.org/abs/1911.10194>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        in_channels (int): number of input channels from the backbone
        feature_key (str): names of input feature from backbone
        low_level_channels (list): a list of channels of low-level features
        low_level_key (list): a list of name of low-level features used in decoder
        low_level_channels_project (list): a list of channels of low-level features after projection in decoder
        decoder_channels (int): number of channels in decoder
        atrous_rates (tuple): atrous rates for ASPP
        num_classes (int): number of classes
        semantic_loss (nn.Module): loss function
        semantic_loss_weight (float): loss weight
        center_loss (nn.Module): loss function
        center_loss_weight (float): loss weight
        offset_loss (nn.Module): loss function
        offset_loss_weight (float): loss weight
        **kwargs: arguments for instance head
    """

    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes, semantic_loss, semantic_loss_weight, center_loss, center_loss_weight, offset_loss, offset_loss_weight, **kwargs):
        decoder = PanopticDeepLabDecoder(in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project, decoder_channels, atrous_rates, num_classes, **kwargs)
        super(PanopticDeepLab, self).__init__(backbone, decoder)
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()
        self.loss_meter_dict['Semantic loss'] = AverageMeter()
        if kwargs.get('has_instance', False):
            self.center_loss = center_loss
            self.center_loss_weight = center_loss_weight
            self.offset_loss = offset_loss
            self.offset_loss_weight = offset_loss_weight
            self.loss_meter_dict['Center loss'] = AverageMeter()
            self.loss_meter_dict['Offset loss'] = AverageMeter()
        else:
            self.center_loss = None
            self.center_loss_weight = 0
            self.offset_loss = None
            self.offset_loss_weight = 0
        self._init_params()

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        loss = 0
        if targets is not None:
            if 'semantic_weights' in targets.keys():
                semantic_loss = self.semantic_loss(results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
            else:
                semantic_loss = self.semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            loss += semantic_loss
            if self.center_loss is not None:
                center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
                center_loss = self.center_loss(results['center'], targets['center']) * center_loss_weights
                if center_loss_weights.sum() > 0:
                    center_loss = center_loss.sum() / center_loss_weights.sum() * self.center_loss_weight
                else:
                    center_loss = center_loss.sum() * 0
                self.loss_meter_dict['Center loss'].update(center_loss.detach().cpu().item(), batch_size)
                loss += center_loss
            if self.offset_loss is not None:
                offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
                offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
                if offset_loss_weights.sum() > 0:
                    offset_loss = offset_loss.sum() / offset_loss_weights.sum() * self.offset_loss_weight
                else:
                    offset_loss = offset_loss.sum() * 0
                self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
                loss += offset_loss
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)
        return results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MNASNet,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Xception65,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (XceptionBlock,
     lambda: ([], {'channel_list': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_InvertedResidual,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1, 'expansion_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bowenc0221_panoptic_deeplab(_paritybench_base):
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

