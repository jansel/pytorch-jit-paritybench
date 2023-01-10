import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cityscapes = _module
utils = _module
voc = _module
main = _module
metrics = _module
stream_metrics = _module
network = _module
_deeplab = _module
backbone = _module
hrnetv2 = _module
mobilenetv2 = _module
resnet = _module
xception = _module
modeling = _module
utils = _module
predict = _module
ext_transforms = _module
loss = _module
scheduler = _module
utils = _module
visualizer = _module

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


from collections import namedtuple


import torch


import torch.utils.data as data


import numpy as np


import collections


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import check_integrity


import random


from torch.utils import data


import torch.nn as nn


import matplotlib


import matplotlib.pyplot as plt


from torch import nn


from torch.nn import functional as F


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


from torch.nn import init


from collections import OrderedDict


from torch.utils.data import dataset


from torchvision import transforms as T


import torchvision


import torchvision.transforms.functional as F


import numbers


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import StepLR


from torchvision.transforms.functional import normalize


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.classifier = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):

    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        self.classifier = nn.Sequential(ASPP(in_channels, aspp_dilate), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias))
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


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


class StageModule(nn.Module):

    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()
        self.number_of_branches = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        for i in range(self.number_of_branches):
            channels = c * 2 ** i
            branch = nn.Sequential(*[BasicBlock(channels, channels) for _ in range(4)])
            self.branches.append(branch)
        self.fuse_layers = nn.ModuleList()
        for branch_output_number in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for branch_number in range(self.number_of_branches):
                if branch_number == branch_output_number:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif branch_number > branch_output_number:
                    self.fuse_layers[-1].append(nn.Sequential(nn.Conv2d(c * 2 ** branch_number, c * 2 ** branch_output_number, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(c * 2 ** branch_output_number, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Upsample(scale_factor=2.0 ** (branch_number - branch_output_number), mode='nearest')))
                elif branch_number < branch_output_number:
                    downsampling_fusion = []
                    for _ in range(branch_output_number - branch_number - 1):
                        downsampling_fusion.append(nn.Sequential(nn.Conv2d(c * 2 ** branch_number, c * 2 ** branch_number, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c * 2 ** branch_number, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))
                    downsampling_fusion.append(nn.Sequential(nn.Conv2d(c * 2 ** branch_number, c * 2 ** branch_output_number, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c * 2 ** branch_output_number, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                    self.fuse_layers[-1].append(nn.Sequential(*downsampling_fusion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [branch(branch_input) for branch, branch_input in zip(self.branches, x)]
        x_fused = []
        for branch_output_index in range(self.output_branches):
            for input_index in range(self.number_of_branches):
                if input_index == 0:
                    x_fused.append(self.fuse_layers[branch_output_index][input_index](x[input_index]))
                else:
                    x_fused[branch_output_index] = x_fused[branch_output_index] + self.fuse_layers[branch_output_index][input_index](x[input_index])
        for i in range(self.output_branches):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused


class HRNet(nn.Module):

    def __init__(self, c=48, num_blocks=[1, 4, 3], num_classes=1000):
        super(HRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(256, eps=1e-05, affine=True, track_running_stats=True))
        bn_expansion = Bottleneck.expansion
        self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample=downsample), Bottleneck(bn_expansion * 64, 64), Bottleneck(bn_expansion * 64, 64), Bottleneck(bn_expansion * 64, 64))
        self.transition1 = nn.ModuleList([nn.Sequential(nn.Conv2d(256, c, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(c, eps=1e-05, affine=True, track_running_stats=True), nn.ReLU(inplace=True)), nn.Sequential(nn.Sequential(nn.Conv2d(256, c * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c * 2, eps=1e-05, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        number_blocks_stage2 = num_blocks[0]
        self.stage2 = nn.Sequential(*[StageModule(stage=2, output_branches=2, c=c) for _ in range(number_blocks_stage2)])
        self.transition2 = self._make_transition_layers(c, transition_number=2)
        number_blocks_stage3 = num_blocks[1]
        self.stage3 = nn.Sequential(*[StageModule(stage=3, output_branches=3, c=c) for _ in range(number_blocks_stage3)])
        self.transition3 = self._make_transition_layers(c, transition_number=3)
        number_blocks_stage4 = num_blocks[2]
        self.stage4 = nn.Sequential(*[StageModule(stage=4, output_branches=4, c=c) for _ in range(number_blocks_stage4)])
        out_channels = sum([(c * 2 ** i) for i in range(len(num_blocks) + 1)])
        pool_feature_map = 8
        self.bn_classifier = nn.Sequential(nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels // 4, eps=1e-05, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(pool_feature_map), nn.Flatten(), nn.Linear(pool_feature_map * pool_feature_map * (out_channels // 4), num_classes))

    @staticmethod
    def _make_transition_layers(c, transition_number):
        return nn.Sequential(nn.Conv2d(c * 2 ** (transition_number - 1), c * 2 ** transition_number, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c * 2 ** transition_number, eps=1e-05, affine=True, track_running_stats=True), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x.append(self.transition2(x[-1]))
        x = self.stage3(x)
        x.append(self.transition3(x[-1]))
        x = self.stage4(x)
        output_h, output_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], dim=1)
        x = self.bn_classifier(x)
        return x


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


def fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end, pad_beg, pad_end


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
        self.input_padding = fixed_padding(3, dilation)

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


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

    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation = 1
        previous_dilation = 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
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

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, dilation=1):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=dilation, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=dilation, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=dilation, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, replace_stride_with_dilation=None):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError('replace_stride_with_dilation should be None or a 4-element tuple, got {}'.format(replace_stride_with_dilation))
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1 = self._make_block(64, 128, 2, 2, start_with_relu=False, grow_first=True, dilate=replace_stride_with_dilation[0])
        self.block2 = self._make_block(128, 256, 2, 2, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[1])
        self.block3 = self._make_block(256, 728, 2, 2, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block4 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block5 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block6 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block7 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block8 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block9 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block10 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block11 = self._make_block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block12 = self._make_block(728, 1024, 2, 2, start_with_relu=True, grow_first=False, dilate=replace_stride_with_dilation[3])
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1, dilation=self.dilation)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1, dilation=self.dilation)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

    def _make_block(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, dilate=False):
        if dilate:
            self.dilation *= strides
            strides = 1
        return Block(in_filters, out_filters, reps, strides, start_with_relu=start_with_relu, grow_first=grow_first, dilation=self.dilation)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class _SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        self.hrnet_flag = hrnet_flag
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'):
                if name == 'transition1':
                    x = [trans(x) for trans in module]
                else:
                    x.append(module(x[-1]))
            else:
                x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag:
                    output_h, output_w = x[0].size(2), x[0].size(3)
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AtrousSeparableConvolution,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'dilation': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_VainF_DeepLabV3Plus_Pytorch(_paritybench_base):
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

