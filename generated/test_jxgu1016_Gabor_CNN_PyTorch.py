import sys
_module = sys.modules[__name__]
del sys
main = _module
net_factory = _module
utils = _module
gcn = _module
GConv = _module
layers = _module
gradtest = _module
setup = _module

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


from torchvision import datasets


from torchvision import transforms


import torch.optim as optim


import torch.nn as nn


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import MultiStepLR


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import math


from torch import nn


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.nn.modules.conv import _ConvNd


from torch.autograd.function import once_differentiable


from torch.autograd import gradcheck


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


class GOF_Function(Function):

    @staticmethod
    def forward(ctx, weight, gaborFilterBank):
        ctx.save_for_backward(weight, gaborFilterBank)
        output = _C.gof_forward(weight, gaborFilterBank)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        weight, gaborFilterBank = ctx.saved_tensors
        grad_weight = _C.gof_backward(grad_output, gaborFilterBank)
        return grad_weight, None


class MConv(_ConvNd):
    """
    Baee layer class for modulated convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, M=4, nScale=3, stride=1, padding=0, dilation=1, groups=1, bias=True, expand=False, padding_mode='zeros'):
        if groups != 1:
            raise ValueError('Group-conv not supported!')
        kernel_size = (M,) + _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.expand = expand
        self.M = M
        self.need_bias = bias
        self.generate_MFilters(nScale, kernel_size)
        self.GOF_Function = GOF_Function.apply

    def generate_MFilters(self, nScale, kernel_size):
        raise NotImplementedError

    def forward(self, x):
        if self.expand:
            x = self.do_expanding(x)
        new_weight = self.GOF_Function(self.weight, self.MFilters)
        new_bias = self.expand_bias(self.bias) if self.need_bias else self.bias
        if self.padding_mode == 'circular':
            expanded_padding = (self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'), self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(x, new_weight, new_bias, self.stride, self.padding, self.dilation, self.groups)

    def do_expanding(self, x):
        index = []
        for i in range(x.size(1)):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index) if x.is_cuda else torch.LongTensor(index)
        return x.index_select(1, index)

    def expand_bias(self, bias):
        index = []
        for i in range(bias.size()):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index) if bias.is_cuda else torch.LongTensor(index)
        return bias.index_select(0, index)


def getGaborFilterBank(nScale, M, h, w):
    Kmax = math.pi / 2
    f = math.sqrt(2)
    sigma = math.pi
    sqsigma = sigma ** 2
    postmean = math.exp(-sqsigma / 2)
    if h != 1:
        gfilter_real = torch.zeros(M, h, w)
        for i in range(M):
            theta = i / M * math.pi
            k = Kmax / f ** (nScale - 1)
            xymax = -1e1000
            xymin = 1e1000
            for y in range(h):
                for x in range(w):
                    y1 = y + 1 - (h + 1) / 2
                    x1 = x + 1 - (w + 1) / 2
                    tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
                    tmp2 = math.cos(k * math.cos(theta) * x1 + k * math.sin(theta) * y1) - postmean
                    gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma
                    xymax = max(xymax, gfilter_real[i][y][x])
                    xymin = min(xymin, gfilter_real[i][y][x])
            gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
    else:
        gfilter_real = torch.ones(M, h, w)
    return gfilter_real


class GConv(MConv):
    """
    Gabor Convolutional Operation Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, M=4, nScale=3, stride=1, padding=0, dilation=1, groups=1, bias=True, expand=False, padding_mode='zeros'):
        super(GConv, self).__init__(in_channels, out_channels, kernel_size, M, nScale, stride, padding, dilation, groups, bias, expand, padding_mode)

    def generate_MFilters(self, nScale, kernel_size):
        self.register_buffer('MFilters', getGaborFilterBank(nScale, *kernel_size))


class GCN(nn.Module):

    def __init__(self, channel=4):
        super(GCN, self).__init__()
        self.channel = channel
        self.model = nn.Sequential(GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True), nn.BatchNorm2d(10 * channel), nn.ReLU(inplace=True), GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False), nn.BatchNorm2d(20 * channel), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False), nn.BatchNorm2d(40 * channel), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False), nn.BatchNorm2d(80 * channel), nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(80, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 80, self.channel)
        x = torch.max(x, 2)[0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

