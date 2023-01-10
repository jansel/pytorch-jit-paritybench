import sys
_module = sys.modules[__name__]
del sys
example = _module
ic_res101_k13_coco_640x640 = _module
ic_res50_k13_coco_640x640 = _module
res101_coco_640x640 = _module
res50_coco_640x640 = _module
ic_conv2d = _module
ic_resnet = _module
ic_conv2d = _module
ic_resnet = _module

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


import re


import copy


import torch.utils.checkpoint as cp


import math


class ICConv2d(nn.Module):

    def __init__(self, pattern_dist, inplanes, planes, kernel_size, stride=1, groups=1, bias=False):
        super(ICConv2d, self).__init__()
        self.conv_list = nn.ModuleList()
        self.planes = planes
        for pattern in pattern_dist:
            channel = pattern_dist[pattern]
            pattern_trans = re.findall('\\d+\\.?\\d*', pattern)
            pattern_trans[0] = int(pattern_trans[0]) + 1
            pattern_trans[1] = int(pattern_trans[1]) + 1
            if channel > 0:
                padding = [0, 0]
                padding[0] = (kernel_size + 2 * (pattern_trans[0] - 1)) // 2
                padding[1] = (kernel_size + 2 * (pattern_trans[1] - 1)) // 2
                self.conv_list.append(nn.Conv2d(inplanes, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups, dilation=pattern_trans))

    def forward(self, x):
        out = []
        for conv in self.conv_list:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        assert out.shape[1] == self.planes
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(BasicBlock, self).__init__()
        global pattern, pattern_index
        pattern_index = pattern_index + 1
        self.conv1 = ICConv2d(pattern[pattern_index], inplanes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        pattern_index = pattern_index + 1
        self.conv2 = ICConv2d(pattern[pattern_index], planes, planes, kernel_size=kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(Bottleneck, self).__init__()
        global pattern, pattern_index
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        pattern_index = pattern_index + 1
        self.conv2 = ICConv2d(pattern[pattern_index], planes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self, block, num_blocks, in_channels, out_channels, expansion=None, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), downsample_first=True, **kwargs):
        norm_cfg = copy.deepcopy(norm_cfg)
        self.block = block
        self.expansion = get_expansion(block, expansion)
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, out_channels)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        if downsample_first:
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        else:
            for i in range(0, num_blocks - 1):
                layers.append(block(in_channels=in_channels, out_channels=in_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super().__init__(*layers)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, kernel_size=3, pattern_path=None):
        super(ResNet, self).__init__()
        global pattern
        with open(pattern_path, 'r') as fin:
            pattern = json.load(fin)
        self.inplanes = 64
        self.kernel_size = kernel_size
        global pattern_index
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()
        assert len(pattern) == pattern_index + 1

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=self.kernel_size))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

