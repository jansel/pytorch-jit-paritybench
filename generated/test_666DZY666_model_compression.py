import sys
_module = sys.modules[__name__]
del sys
gc_prune = _module
main = _module
models = _module
nin = _module
nin_gc = _module
normal_regular_prune = _module
bn_folding = _module
bn_folding_model_test = _module
nin_gc_inference = _module
nin_gc_training = _module
util_wt_bab = _module
main = _module
nin = _module
nin_bn_conv = _module
nin_gc = _module
util_wt_bab = _module
main = _module
nin = _module
nin_gc = _module
util_wqaq = _module
main = _module
nin = _module
nin_gc = _module
util_wqaq = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


from torch.autograd import Variable


from torchvision import datasets


from torchvision import transforms


import numpy as np


import math


import torch.optim as optim


import torchvision


import torchvision.transforms as transforms


import torch.nn.functional as F


import time


from torch.autograd import Function


from torch import distributed


from torch.nn import init


from torch.nn.parameter import Parameter


def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, channels, height, width)
    return x


class FP_Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, channel_shuffle=0, shuffle_groups=1, last=0, bin_mp=0, bin_nor=1, last_relu=0, first=0):
        super(FP_Conv2d, self).__init__()
        self.dropout_ratio = dropout
        self.last = last
        self.first_flag = first
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.first_flag:
            x = self.relu(x)
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class activation_quantize(nn.Module):

    def __init__(self, a_bits):
        super().__init__()
        self.a_bits = a_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            None
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)
            scale = float(2 ** self.a_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
        return output


class weight_quantize(nn.Module):

    def __init__(self, w_bits):
        super().__init__()
        self.w_bits = w_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            None
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5
            scale = float(2 ** self.w_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
            output = 2 * output - 1
        return output


class Conv2d_Q(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, a_bits=8, w_bits=8, first_layer=0):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)
        self.first_layer = first_layer

    def forward(self, input):
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight)
        output = F.conv2d(input=q_input, weight=q_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output


class DorefaConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, channel_shuffle=0, shuffle_groups=1, last_relu=0, abits=8, wbits=8, first_layer=0):
        super(DorefaConv2d, self).__init__()
        self.last_relu = last_relu
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.first_layer = first_layer
        self.q_conv = Conv2d_Q(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, first_layer=first_layer)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.first_layer:
            x = self.relu(x)
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.q_conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self, cfg=None, abits=8, wbits=8):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]
        self.dorefa = nn.Sequential(DorefaConv2d(3, cfg[0], kernel_size=5, stride=1, padding=2, abits=abits, wbits=wbits, first_layer=1), DorefaConv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=0, abits=abits, wbits=wbits), DorefaConv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2, abits=abits, wbits=wbits), nn.MaxPool2d(kernel_size=2, stride=2, padding=0), DorefaConv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2, abits=abits, wbits=wbits), DorefaConv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16, abits=abits, wbits=wbits), DorefaConv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4, abits=abits, wbits=wbits), nn.MaxPool2d(kernel_size=2, stride=2, padding=0), DorefaConv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4, abits=abits, wbits=wbits), DorefaConv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32, abits=abits, wbits=wbits), DorefaConv2d(cfg[7], 10, kernel_size=1, stride=1, padding=0, last_relu=1, abits=abits, wbits=wbits), nn.AvgPool2d(kernel_size=8, stride=1, padding=0))

    def forward(self, x):
        x = self.dorefa(x)
        x = x.view(x.size(0), -1)
        return x


class DummyModule(nn.Module):

    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


class Binary_a(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        """
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        """
        return grad_input


class activation_bin(nn.Module):

    def __init__(self, A):
        super().__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = Binary_a.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
        else:
            output = self.relu(input)
        return output


class Tnn_Bin_Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, channel_shuffle=0, shuffle_groups=1, A=2, W=2, last_relu=0, last_bin=0):
        super(Tnn_Bin_Conv2d, self).__init__()
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.last_relu = last_relu
        self.last_bin = last_bin
        self.tnn_bin_conv = Conv2d_Q(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, A=A, W=W)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bin_a = activation_bin(A=A)

    def forward(self, x):
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.tnn_bin_conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        if self.last_bin:
            x = self.bin_a(x)
        return x


class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Ternary(Function):

    @staticmethod
    def forward(self, input):
        E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        threshold = E * 0.7
        output = torch.sign(torch.add(torch.sign(torch.add(input, threshold)), torch.sign(torch.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        grad_input = grad_output.clone()
        return grad_input


def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean)
    w.data.clamp(-1.0, 1.0)
    return w


class weight_tnn_bin(nn.Module):

    def __init__(self, W):
        super().__init__()
        self.W = W

    def binary(self, input):
        output = Binary_w.apply(input)
        return output

    def ternary(self, input):
        output = Ternary.apply(input)
        return output

    def forward(self, input):
        if self.W == 2 or self.W == 3:
            if self.W == 2:
                output = meancenter_clampConvParams(input)
                E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
                alpha = E
                output = self.binary(output)
                output = output * alpha
            elif self.W == 3:
                output_fp = input.clone()
                output, threshold = self.ternary(input)
                output_abs = torch.abs(output_fp)
                mask_le = output_abs.le(threshold)
                mask_gt = output_abs.gt(threshold)
                output_abs[mask_le] = 0
                output_abs_th = output_abs.clone()
                output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
                mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                alpha = output_abs_th_sum / mask_gt_sum
                output = output * alpha
        else:
            output = input
        return output


class Quantizer(nn.Module):

    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

    def update_params(self):
        raise NotImplementedError

    def quantize(self, input):
        output = input * self.scale - self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    def clamp(self, input):
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    def dequantize(self, input):
        output = (input + self.zero_point) / self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            None
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.update_params()
            output = self.quantize(input)
            output = self.round(output)
            output = self.clamp(output)
            output = self.dequantize(output)
        return output


class UnsignedQuantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))


class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = quantized_range / float_range
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)


class RangeTracker(nn.Module):

    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
        self.update_range(min_val, max_val)


class AveragedRangeTracker(RangeTracker):

    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)


class GlobalRangeTracker(RangeTracker):

    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))


class SignedQuantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << self.bits - 1)))
        self.register_buffer('max_val', torch.tensor((1 << self.bits - 1) - 1))


class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))
        self.scale = quantized_range / float_range
        self.zero_point = torch.zeros_like(self.scale)


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


class BNFold_Conv2d_Q(Conv2d_Q):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, eps=1e-05, momentum=0.01, a_bits=8, w_bits=8, q_type=1, first_layer=0):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        if self.training:
            output = F.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        else:
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(weight)
        if self.training:
            output = F.conv2d(input=q_input, weight=q_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
            output += reshape_to_activation(bias)
        else:
            output = F.conv2d(input=q_input, weight=q_weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output


class QuanConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, channel_shuffle=0, shuffle_groups=1, last_relu=0, abits=8, wbits=8, bn_fold=0, q_type=1, first_layer=0):
        super(QuanConv2d, self).__init__()
        self.last_relu = last_relu
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.bn_fold = bn_fold
        self.first_layer = first_layer
        if self.bn_fold == 1:
            self.bn_q_conv = BNFold_Conv2d_Q(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, q_type=q_type, first_layer=first_layer)
        else:
            self.q_conv = Conv2d_Q(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, q_type=q_type, first_layer=first_layer)
            self.bn = nn.BatchNorm2d(output_channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.first_layer:
            x = self.relu(x)
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        if self.bn_fold == 1:
            x = self.bn_q_conv(x)
        else:
            x = self.q_conv(x)
            x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class Linear_Q(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)

    def forward(self, input):
        q_input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(self.weight)
        output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNFold_Conv2d_Q,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2d_Q,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear_Q,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (activation_bin,
     lambda: ([], {'A': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (activation_quantize,
     lambda: ([], {'a_bits': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (weight_quantize,
     lambda: ([], {'w_bits': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (weight_tnn_bin,
     lambda: ([], {'W': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_666DZY666_model_compression(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

