import sys
_module = sys.modules[__name__]
del sys
arch = _module
base_generator = _module
regnet = _module
generate_configs = _module
prepare_imagenet = _module
search = _module
test_flops = _module
train = _module
verify = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn as nn


import copy


import random


import logging


from torchvision import transforms


import time


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.backends.cudnn as cudnn


from torch.nn.parallel import DistributedDataParallel


class AnyNeSt(nn.Module):

    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio,
        ls_group_width, stride):
        super().__init__()
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width,
            ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module('stem', ConvBnAct(3, prev_block_width,
            kernel_size=3, stride=2, padding=1, bias=False))
        for i, (num_blocks, block_width, bottleneck_ratio, group_width
            ) in enumerate(zip(ls_num_blocks, ls_block_width,
            ls_bottleneck_ratio, ls_group_width)):
            self.net.add_module('stage_{}'.format(i), Stage(num_blocks,
                prev_block_width, block_width, bottleneck_ratio,
                group_width=group_width, stride=stride))
            prev_block_width = block_width
        self.net.add_module('pool', GlobalAvgPool2d())
        self.net.add_module('fc', nn.Linear(ls_block_width[-1], 1000))

    def forward(self, x):
        x = self.net(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_ratio,
        group_width, stride):
        super(Bottleneck, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width
        self.conv1 = ConvBnAct(in_channels, inter_channels, kernel_size=1,
            bias=False)
        self.conv2 = ConvBnAct(inter_channels, inter_channels, kernel_size=
            3, stride=stride, groups=groups, padding=1, bias=False)
        self.conv3 = ConvBnAct(inter_channels, out_channels, kernel_size=1,
            bias=False, act=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBnAct(in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False, act=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.relu(x1 + x2)
        return x


class Stage(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels,
        bottleneck_ratio, group_width, stride):
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module('block_0', Bottleneck(in_channels,
            out_channels, bottleneck_ratio, group_width, stride=stride))
        for i in range(1, num_blocks):
            self.blocks.add_module('block_{}'.format(i), Bottleneck(
                out_channels, out_channels, bottleneck_ratio, group_width,
                stride=1))

    def forward(self, x):
        x = self.blocks(x)
        return x


class ConvBnAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        act=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=
            padding_mode))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        if act:
            self.add_module('relu', nn.ReLU())


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.
            size(0), -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhanghang1989_RegNet_Search_PyTorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvBnAct(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

