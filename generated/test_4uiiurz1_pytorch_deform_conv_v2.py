import sys
_module = sys.modules[__name__]
del sys
deform_conv_v2 = _module
archs = _module
dataset = _module
scaled_mnist_train = _module
utils = _module

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


import numpy as np


from torch.nn import functional as F


from torchvision import models


import torchvision


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import Dataset


from matplotlib import pyplot as plt


import random


import scipy.ndimage as ndi


import pandas as pd


from collections import OrderedDict


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.optim as optim


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import math


class DeformConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[(...), :N], 0, x.size(2) - 1), torch.clamp(q_lt[(...), N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[(...), :N], 0, x.size(2) - 1), torch.clamp(q_rb[(...), N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[(...), :N], q_rb[(...), N:]], dim=-1)
        q_rt = torch.cat([q_rb[(...), :N], q_lt[(...), N:]], dim=-1)
        p = torch.cat([torch.clamp(p[(...), :N], 0, x.size(2) - 1), torch.clamp(p[(...), N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[(...), :N].type_as(p) - p[(...), :N])) * (1 + (q_lt[(...), N:].type_as(p) - p[(...), N:]))
        g_rb = (1 - (q_rb[(...), :N].type_as(p) - p[(...), :N])) * (1 - (q_rb[(...), N:].type_as(p) - p[(...), N:]))
        g_lb = (1 + (q_lb[(...), :N].type_as(p) - p[(...), :N])) * (1 - (q_lb[(...), N:].type_as(p) - p[(...), N:]))
        g_rt = (1 - (q_rt[(...), :N].type_as(p) - p[(...), :N])) * (1 + (q_rt[(...), N:].type_as(p) - p[(...), N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[(...), :N] * padded_w + q[(...), N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[(...), s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


class ScaledMNISTNet(nn.Module):

    def __init__(self, args, num_classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        features = []
        inplanes = 1
        outplanes = 32
        for i in range(4):
            if args.deform and args.min_deform_layer <= i + 1:
                features.append(DeformConv2d(inplanes, outplanes, 3, padding=1, bias=False, modulation=args.modulation))
            else:
                features.append(nn.Conv2d(inplanes, outplanes, 3, padding=1, bias=False))
            features.append(nn.BatchNorm2d(outplanes))
            features.append(self.relu)
            if i == 1:
                features.append(self.pool)
            inplanes = outplanes
            outplanes *= 2
        self.features = nn.Sequential(*features)
        self.fc = nn.Linear(256, 10)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeformConv2d,
     lambda: ([], {'inc': 4, 'outc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledMNISTNet,
     lambda: ([], {'args': _mock_config(deform=4, min_deform_layer=1, modulation=4), 'num_classes': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
]

class Test_4uiiurz1_pytorch_deform_conv_v2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

