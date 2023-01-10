import sys
_module = sys.modules[__name__]
del sys
cifar_train_eval = _module
imgnet_train_eval = _module
cifar_resnet = _module
imgnet_alexnet = _module
preprocessing = _module
quant_dorefa = _module

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


import time


import torch


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision


import torch.utils.data


import torchvision.datasets as datasets


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import numpy as np


def uniform_quantize(k):


    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return qfn().apply


class activation_quantize_fn(nn.Module):

    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


class weight_quantize_fn(nn.Module):

    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        return weight_q


def conv2d_Q_fn(w_bit):


    class Conv2d_Q(nn.Conv2d):

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return Conv2d_Q


class PreActBlock_conv_Q(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(self, wbit, abit, in_planes, out_planes, stride=1):
        super(PreActBlock_conv_Q, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        self.act_q = activation_quantize_fn(a_bit=abit)
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_conv = None
        if stride != 1:
            self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.act_q(F.relu(self.bn0(x)))
        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x
        out = self.conv0(out)
        out = self.act_q(F.relu(self.bn1(out)))
        out = self.conv1(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_units, wbit, abit, num_classes):
        super(PreActResNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.ModuleList()
        in_planes = 16
        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
        for stride, channel in zip(strides, channels):
            self.layers.append(block(wbit, abit, in_planes, channel, stride))
            in_planes = channel
        self.bn = nn.BatchNorm2d(64)
        self.logit = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.logit(out)
        return out


def linear_Q_fn(w_bit):


    class Linear_Q(nn.Linear):

        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)
    return Linear_Q


class AlexNet_Q(nn.Module):

    def __init__(self, wbit, abit, num_classes=1000):
        super(AlexNet_Q, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        Linear = linear_Q_fn(w_bit=wbit)
        self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(96), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True), Conv2d(96, 256, kernel_size=5, padding=2), nn.BatchNorm2d(256), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit), Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit), Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit), Conv2d(384, 256, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit))
        self.classifier = nn.Sequential(Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit), Linear(4096, 4096), nn.ReLU(inplace=True), activation_quantize_fn(a_bit=abit), nn.Linear(4096, num_classes))
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PreActBlock_conv_Q,
     lambda: ([], {'wbit': 4, 'abit': 4, 'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (activation_quantize_fn,
     lambda: ([], {'a_bit': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (weight_quantize_fn,
     lambda: ([], {'w_bit': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_zzzxxxttt_pytorch_DoReFaNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

