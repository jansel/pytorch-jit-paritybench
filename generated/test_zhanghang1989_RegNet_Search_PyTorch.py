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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


class ConvBnAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', act=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        if act:
            self.add_module('relu', nn.ReLU())


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        super(Bottleneck, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width
        self.conv1 = ConvBnAct(in_channels, inter_channels, kernel_size=1, bias=False)
        self.conv2 = ConvBnAct(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.conv3 = ConvBnAct(inter_channels, out_channels, kernel_size=1, bias=False, act=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBnAct(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, act=False)
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

    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module('block_0', Bottleneck(in_channels, out_channels, bottleneck_ratio, group_width, stride=stride))
        for i in range(1, num_blocks):
            self.blocks.add_module('block_{}'.format(i), Bottleneck(out_channels, out_channels, bottleneck_ratio, group_width, stride=1))

    def forward(self, x):
        x = self.blocks(x)
        return x


class AnyNeSt(nn.Module):

    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super().__init__()
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module('stem', ConvBnAct(3, prev_block_width, kernel_size=3, stride=2, padding=1, bias=False))
        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)):
            self.net.add_module('stage_{}'.format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width=group_width, stride=stride))
            prev_block_width = block_width
        self.net.add_module('pool', GlobalAvgPool2d())
        self.net.add_module('fc', nn.Linear(ls_block_width[-1], 1000))

    def forward(self, x):
        x = self.net(x)
        return x


class RegNet(AnyNeSt):

    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride=2):
        parameterized_width = initial_width + slope * np.arange(network_depth)
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        ls_group_width = ls_group_width.tolist()
        ls_block_width = ls_block_width.astype(np.int).tolist()
        super().__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride=stride)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bottleneck,
     lambda: ([], {'in_channels': 256, 'out_channels': 16, 'bottleneck_ratio': 4, 'group_width': 4, 'stride': 256}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (ConvBnAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stage,
     lambda: ([], {'num_blocks': 4, 'in_channels': 256, 'out_channels': 16, 'bottleneck_ratio': 4, 'group_width': 4, 'stride': 256}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
]

class Test_zhanghang1989_RegNet_Search_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

