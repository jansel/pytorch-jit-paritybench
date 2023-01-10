import sys
_module = sys.modules[__name__]
del sys
function = _module
setup = _module
denoise = _module
function = _module
module = _module
similar = _module
weighting = _module

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


from torch import nn


import torch.nn.functional as F


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch import cuda


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import random


import math


import torchvision


import torchvision.transforms as transforms


import copy


import time


from torch.nn import functional as F


class similarFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = kH, kW
        output = similar_forward(x_ori, x_loc, kH, kW)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, False)
        return grad_ori, grad_loc, None, None


f_similar = similarFunction.apply


class weightingFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = kH, kW
        output = weighting_forward(x_ori, x_weight, kH, kW)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW)
        return grad_ori, grad_weight, None, None


f_weighting = weightingFunction.apply


class LocalAttention(nn.Module):

    def __init__(self, inp_channels, out_channels, kH, kW):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        weight = f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = f_weighting(x3, weight, self.kH, self.kW)
        return out


class TorchLocalAttention(nn.Module):

    def __init__(self, inp_channels, out_channels, kH, kW):
        super(TorchLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW

    @staticmethod
    def f_similar(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()
        pad = kh // 2, kw // 2
        x_theta = x_theta.permute(0, 2, 3, 1).contiguous()
        x_theta = x_theta.view(n * h * w, 1, c)
        x_phi = F.unfold(x_phi, kernel_size=(kh, kw), stride=1, padding=pad)
        x_phi = x_phi.contiguous().view(n, c, kh * kw, h * w)
        x_phi = x_phi.permute(0, 3, 1, 2).contiguous()
        x_phi = x_phi.view(n * h * w, c, kh * kw)
        out = torch.matmul(x_theta, x_phi)
        out = out.view(n, h, w, kh * kw)
        return out

    @staticmethod
    def f_weighting(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()
        pad = kh // 2, kw // 2
        x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad)
        x_theta = x_theta.permute(0, 2, 1).contiguous()
        x_theta = x_theta.view(n * h * w, c, kh * kw)
        x_phi = x_phi.view(n * h * w, kh * kw, 1)
        out = torch.matmul(x_theta, x_phi)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        weight = self.f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(x3, weight, self.kH, self.kW)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        if args.mode == 'torch':
            self.main = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.ReLU(True), TorchLocalAttention(64, 64, 5, 5), nn.ReLU(True), TorchLocalAttention(64, 64, 5, 5), nn.ReLU(True), TorchLocalAttention(64, 64, 5, 5), nn.ReLU(True), nn.Conv2d(64, 3, 3, padding=1, bias=False))
        else:
            self.main = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.ReLU(True), LocalAttention(64, 64, 5, 5), nn.ReLU(True), LocalAttention(64, 64, 5, 5), nn.ReLU(True), LocalAttention(64, 64, 5, 5), nn.ReLU(True), nn.Conv2d(64, 3, 3, padding=1, bias=False))

    def forward(self, x):
        out = self.main(x)
        return out + x

