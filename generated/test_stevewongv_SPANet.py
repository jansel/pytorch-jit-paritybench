import sys
_module = sys.modules[__name__]
del sys
SPANet = _module
cal_ssim = _module
dataset = _module
irnn = _module
main = _module
randomcrop = _module

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


from collections import OrderedDict


from torch.autograd import Variable


import numpy as np


from math import exp


from numpy.random import RandomState


from torch.utils.data import Dataset


import math


import logging


from torch.nn import MSELoss


from torch.optim import Adam


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils.data import DataLoader


import random


import numbers


import torchvision.transforms.functional as F


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class irnn(torch.autograd.Function):

    def __init__(self):
        super(irnn, self).__init__()

    def forward(self, input_feature, weight_up, weight_right, weight_down, weight_left, bias_up, bias_right, bias_down, bias_left):
        assert input_feature.is_contiguous() == True
        assert weight_left.is_contiguous() == True
        assert weight_right.is_contiguous() == True
        assert weight_down.is_contiguous() == True
        assert weight_up.is_contiguous() == True
        assert bias_left.is_contiguous() == True
        assert bias_right.is_contiguous() == True
        assert bias_up.is_contiguous() == True
        assert bias_down.is_contiguous() == True
        output_left = input_feature.clone()
        output_right = input_feature.clone()
        output_up = input_feature.clone()
        output_down = input_feature.clone()
        if input_feature.is_cuda == True:
            n = input_feature.nelement()
            cuda_num_threads = 1024
            cunnex('IRNNForward')(grid=tuple([int((n + cuda_num_threads - 1) / cuda_num_threads), 1, 1]), block=tuple([cuda_num_threads, 1, 1]), args=[input_feature.data_ptr(), weight_up.data_ptr(), weight_right.data_ptr(), weight_down.data_ptr(), weight_left.data_ptr(), bias_up.data_ptr(), bias_right.data_ptr(), bias_down.data_ptr(), bias_left.data_ptr(), output_up.data_ptr(), output_right.data_ptr(), output_down.data_ptr(), output_left.data_ptr(), input_feature.size(1), input_feature.size(2), input_feature.size(3), n], stream=Stream)
        elif input_feature.is_cuda == False:
            raise NotImplementedError()
        self.save_for_backward(input_feature, weight_up, weight_right, weight_down, weight_left, output_up, output_right, output_down, output_left)
        return output_up, output_right, output_down, output_left

    def backward(self, grad_output_up, grad_output_right, grad_output_down, grad_output_left):
        input_feature, weight_up, weight_right, weight_down, weight_left, output_up, output_right, output_down, output_left = self.saved_tensors
        if grad_output_up.is_contiguous() != True:
            grad_output_up = grad_output_up.contiguous()
        if grad_output_right.is_contiguous() != True:
            grad_output_right = grad_output_right.contiguous()
        if grad_output_down.is_contiguous() != True:
            grad_output_down = grad_output_down.contiguous()
        if grad_output_left.is_contiguous() != True:
            grad_output_left = grad_output_left.contiguous()
        grad_input = torch.zeros_like(input_feature)
        grad_weight_up_map = torch.zeros_like(input_feature)
        grad_weight_right_map = torch.zeros_like(input_feature)
        grad_weight_down_map = torch.zeros_like(input_feature)
        grad_weight_left_map = torch.zeros_like(input_feature)
        grad_weight_left = torch.zeros_like(weight_left)
        grad_weight_right = torch.zeros_like(weight_left)
        grad_weight_up = torch.zeros_like(weight_left)
        grad_weight_down = torch.zeros_like(weight_left)
        grad_bias_up_map = torch.zeros_like(input_feature)
        grad_bias_right_map = torch.zeros_like(input_feature)
        grad_bias_down_map = torch.zeros_like(input_feature)
        grad_bias_left_map = torch.zeros_like(input_feature)
        if input_feature.is_cuda == True:
            n = grad_input.nelement()
            cuda_num_threads = 1024
            cunnex('IRNNBackward')(grid=tuple([int((n + cuda_num_threads - 1) / cuda_num_threads), 1, 1]), block=tuple([cuda_num_threads, 1, 1]), args=[grad_input.data_ptr(), grad_weight_up_map.data_ptr(), grad_weight_right_map.data_ptr(), grad_weight_down_map.data_ptr(), grad_weight_left_map.data_ptr(), grad_bias_up_map.data_ptr(), grad_bias_right_map.data_ptr(), grad_bias_down_map.data_ptr(), grad_bias_left_map.data_ptr(), weight_up.data_ptr(), weight_right.data_ptr(), weight_down.data_ptr(), weight_left.data_ptr(), grad_output_up.data_ptr(), grad_output_right.data_ptr(), grad_output_down.data_ptr(), grad_output_left.data_ptr(), output_up.data_ptr(), output_right.data_ptr(), output_down.data_ptr(), output_left.data_ptr(), input_feature.size(1), input_feature.size(2), input_feature.size(3), n], stream=Stream)
            grad_bias_up = torch.zeros_like(weight_left).reshape(weight_left.size(0))
            grad_bias_right = torch.zeros_like(weight_left).reshape(weight_left.size(0))
            grad_bias_down = torch.zeros_like(weight_left).reshape(weight_left.size(0))
            grad_bias_left = torch.zeros_like(weight_left).reshape(weight_left.size(0))
            grad_weight_left = grad_weight_left_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
            grad_weight_right = grad_weight_right_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
            grad_weight_up = grad_weight_up_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
            grad_weight_down = grad_weight_down_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
            grad_bias_up = grad_bias_up_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
            grad_bias_right = grad_bias_right_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
            grad_bias_down = grad_bias_down_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
            grad_bias_left = grad_bias_left_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
        elif input_feature.is_cuda == False:
            raise NotImplementedError()
        return grad_input, grad_weight_up, grad_weight_right, grad_weight_down, grad_weight_left, grad_bias_up, grad_bias_right, grad_bias_down, grad_bias_left


class Spacial_IRNN(nn.Module):

    def __init__(self, in_channels, alpha=0.2):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))

    def forward(self, input):
        return irnn()(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight, self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias, self.left_weight.bias)


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class SAM(nn.Module):

    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels)
        self.irnn2 = Spacial_IRNN(self.out_channels)
        self.conv_in = conv3x3(in_channels, in_channels)
        self.conv2 = conv3x3(in_channels * 4, in_channels)
        self.conv3 = conv3x3(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask


class SPANet(nn.Module):

    def __init__(self):
        super(SPANet, self).__init__()
        self.conv_in = nn.Sequential(conv3x3(3, 32), nn.ReLU(True))
        self.SAM1 = SAM(32, 32, 1)
        self.res_block1 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)
        self.res_block7 = Bottleneck(32, 32)
        self.res_block8 = Bottleneck(32, 32)
        self.res_block9 = Bottleneck(32, 32)
        self.res_block10 = Bottleneck(32, 32)
        self.res_block11 = Bottleneck(32, 32)
        self.res_block12 = Bottleneck(32, 32)
        self.res_block13 = Bottleneck(32, 32)
        self.res_block14 = Bottleneck(32, 32)
        self.res_block15 = Bottleneck(32, 32)
        self.res_block16 = Bottleneck(32, 32)
        self.res_block17 = Bottleneck(32, 32)
        self.conv_out = nn.Sequential(conv3x3(32, 3))

    def forward(self, x):
        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
        Attention1 = self.SAM1(out)
        out = F.relu(self.res_block4(out) * Attention1 + out)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)
        Attention2 = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        Attention3 = self.SAM1(out)
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        Attention4 = self.SAM1(out)
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)
        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
        out = self.conv_out(out)
        return Attention1, out


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_stevewongv_SPANet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

