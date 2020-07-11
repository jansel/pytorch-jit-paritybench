import sys
_module = sys.modules[__name__]
del sys
train_net = _module
vovnet = _module
config = _module
mobilenet = _module
vovnet = _module

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


from torch import nn


from torch.nn import BatchNorm2d


from collections import OrderedDict


import torch


import torch.nn as nn


import torch.nn.functional as F


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), FrozenBatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), FrozenBatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), FrozenBatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), FrozenBatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), FrozenBatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


_NORM = False


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/dw_conv3x3'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_channels, bias=False)), ('{}_{}/pw_conv1x1'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)), ('{}_{}/pw_norm'.format(module_name, postfix), get_norm(_NORM, out_channels)), ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True))]


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(OrderedDict(conv1x1(in_channel, stage_ch, '{}_reduction'.format(module_name), '0')))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))
        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (eSEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_youngwanLEE_vovnet_detectron2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

