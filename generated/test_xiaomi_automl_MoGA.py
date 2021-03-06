import sys
_module = sys.modules[__name__]
del sys
accuracy = _module
dataloader = _module
MoGA_A = _module
MoGA_B = _module
MoGA_C = _module
models = _module
verify = _module

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


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, act, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True), act)
        self.fc = nn.Sequential(nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True), Hsigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.fc(y)
        return torch.mul(x, y)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, act, se):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.act = act
        self.se = se
        padding = kernel_size // 2
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        if self.se:
            self.mid_se = SEModule(hidden_dim, act)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.se:
            x = self.mid_se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return inputs + x
        else:
            return x


def classifier(inp, nclass):
    return nn.Linear(inp, nclass)


def conv_before_pooling(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), Hswish())


def conv_head(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, bias=False), Hswish(inplace=True), nn.Dropout2d(0.2))


def separable_conv(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


def stem(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), Hswish())


class MoGaA(nn.Module):

    def __init__(self, n_class=1000, input_size=224):
        super(MoGaA, self).__init__()
        assert input_size % 32 == 0
        mb_config = [[6, 24, 5, 2, 0, 0], [6, 24, 7, 1, 0, 0], [6, 40, 3, 2, 0, 0], [6, 40, 3, 1, 0, 1], [3, 40, 3, 1, 0, 1], [6, 80, 3, 2, 1, 1], [6, 80, 3, 1, 1, 0], [6, 80, 7, 1, 1, 0], [3, 80, 7, 1, 1, 1], [6, 112, 7, 1, 1, 0], [6, 112, 3, 1, 1, 0], [6, 160, 3, 2, 1, 0], [6, 160, 5, 1, 1, 1], [6, 160, 5, 1, 1, 1]]
        first_filter = 16
        second_filter = 16
        second_last_filter = 960
        last_channel = 1280
        self.last_channel = last_channel
        self.stem = stem(3, first_filter, 2)
        self.separable_conv = separable_conv(first_filter, second_filter)
        self.mb_module = list()
        input_channel = second_filter
        for t, c, k, s, a, se in mb_config:
            output_channel = c
            act = nn.ReLU(inplace=True) if a == 0 else Hswish(inplace=True)
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, act=act, se=se != 0))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, second_last_filter)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_head = conv_head(second_last_filter, last_channel)
        self.classifier = classifier(last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = self.global_pooling(x)
        x = self.conv_head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class MoGaB(nn.Module):

    def __init__(self, n_class=1000, input_size=224):
        super(MoGaB, self).__init__()
        assert input_size % 32 == 0
        mb_config = [[3, 24, 3, 2, 0, 0], [3, 24, 3, 1, 0, 0], [6, 40, 7, 2, 0, 0], [3, 40, 3, 1, 0, 0], [6, 40, 5, 1, 0, 0], [6, 80, 3, 2, 1, 1], [6, 80, 5, 1, 1, 1], [3, 80, 3, 1, 1, 0], [6, 80, 7, 1, 1, 1], [6, 112, 7, 1, 1, 0], [3, 112, 5, 1, 1, 0], [6, 160, 7, 2, 1, 1], [6, 160, 7, 1, 1, 1], [6, 160, 3, 1, 1, 1]]
        first_filter = 16
        second_filter = 16
        second_last_filter = 960
        last_channel = 1280
        self.last_channel = last_channel
        self.stem = stem(3, first_filter, 2)
        self.separable_conv = separable_conv(first_filter, second_filter)
        self.mb_module = list()
        input_channel = second_filter
        for t, c, k, s, a, se in mb_config:
            output_channel = c
            act = nn.ReLU(inplace=True) if a == 0 else Hswish(inplace=True)
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, act=act, se=se != 0))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, second_last_filter)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_head = conv_head(second_last_filter, last_channel)
        self.classifier = classifier(last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = self.global_pooling(x)
        x = self.conv_head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class MoGaC(nn.Module):

    def __init__(self, n_class=1000, input_size=224):
        super(MoGaC, self).__init__()
        assert input_size % 32 == 0
        mb_config = [[3, 24, 5, 2, 0, 0], [3, 24, 3, 1, 0, 0], [3, 40, 5, 2, 0, 0], [3, 40, 3, 1, 0, 0], [3, 40, 5, 1, 0, 0], [3, 80, 5, 2, 1, 0], [6, 80, 5, 1, 1, 1], [3, 80, 5, 1, 1, 0], [3, 80, 5, 1, 1, 0], [6, 112, 3, 1, 1, 0], [6, 112, 3, 1, 1, 1], [6, 160, 3, 2, 1, 1], [6, 160, 3, 1, 1, 1], [6, 160, 3, 1, 1, 1]]
        first_filter = 16
        second_filter = 16
        second_last_filter = 960
        last_channel = 1280
        self.last_channel = last_channel
        self.stem = stem(3, first_filter, 2)
        self.separable_conv = separable_conv(first_filter, second_filter)
        self.mb_module = list()
        input_channel = second_filter
        for t, c, k, s, a, se in mb_config:
            output_channel = c
            act = nn.ReLU(inplace=True) if a == 0 else Hswish(inplace=True)
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, act=act, se=se != 0))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, second_last_filter)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_head = conv_head(second_last_filter, last_channel)
        self.classifier = classifier(last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = self.global_pooling(x)
        x = self.conv_head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MoGaA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MoGaB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MoGaC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_xiaomi_automl_MoGA(_paritybench_base):
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

