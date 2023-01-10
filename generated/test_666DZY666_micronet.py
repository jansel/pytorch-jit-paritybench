import sys
_module = sys.modules[__name__]
del sys
micronet = _module
base_module = _module
op = _module
compression = _module
pruning = _module
gc_prune = _module
main = _module
normal_regular_prune = _module
quantization = _module
wbwtab = _module
bn_fuse = _module
bn_fused_model_test = _module
main = _module
quantize = _module
wqaq = _module
dorefa = _module
main = _module
quant_model_para = _module
quant_model_test = _module
quantize = _module
iao = _module
bn_fuse = _module
bn_fused_model_test = _module
main = _module
quantize = _module
deploy = _module
tensorrt = _module
calibrator = _module
eval_trt = _module
models = _module
models_trt = _module
test_trt = _module
util_trt = _module
nin = _module
nin_gc = _module
resnet = _module
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


import torch


import torch.nn as nn


import numpy as np


from torch.autograd import Variable


from torchvision import datasets


from torchvision import transforms


import math


import torch.optim as optim


import torchvision


import torchvision.transforms as transforms


from torch.nn import init


import copy


import time


import torch.nn.functional as F


from torch.autograd import Function


from torch import distributed


from torch.nn.parameter import Parameter


import logging


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, res, shortcut):
        output = res + shortcut
        return output


class Round(Function):

    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        if q_type == 0:
            max_val = torch.max(torch.abs(observer_min_val), torch.abs(observer_max_val))
            min_val = -max_val
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None


class ActivationQuantizer(nn.Module):

    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
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
            scale = 1 / float(2 ** self.a_bits - 1)
            output = self.round(output / scale) * scale
        return output


class WeightQuantizer(nn.Module):

    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
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
            scale = 1 / float(2 ** self.w_bits - 1)
            output = self.round(output / scale) * scale
            output = 2 * output - 1
        return output


class Quantizer(nn.Module):

    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        if self.observer.q_level == 'L':
            self.register_buffer('scale', torch.ones(1, dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros(1, dtype=torch.float32))
        elif self.observer.q_level == 'C':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.q_level == 'FC':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32))

    def update_qparams(self):
        raise NotImplementedError

    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            None
            assert self.bits != 1
        else:
            if not self.qaft:
                if self.training:
                    if not self.union:
                        self.observer(input)
                    self.update_qparams()
            output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point, self.observer.min_val / self.scale - self.zero_point, self.observer.max_val / self.scale - self.zero_point, self.q_type), self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale.clone()
        return output


class UnsignedQuantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:
            self.register_buffer('quant_min_val', torch.tensor(0, dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor((1 << self.bits) - 2, dtype=torch.float32))
        elif self.activation_weight_flag == 1:
            self.register_buffer('quant_min_val', torch.tensor(0, dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor((1 << self.bits) - 1, dtype=torch.float32))
        else:
            None


class AsymmetricQuantizer(UnsignedQuantizer):

    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)
        float_range = self.observer.max_val - self.observer.min_val
        scale = float_range / quant_range
        scale = torch.max(scale, self.eps)
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(torch.abs(self.observer.min_val / scale) + 0.5)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


class HistogramObserver(nn.Module):

    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer('min_val', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('max_val', torch.zeros(1, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        max_val_cur = torch.kthvalue(input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0)[0]
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)


class ObserverBase(nn.Module):

    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == 'FC':
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]
        self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):

    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1, dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros(1, dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):

    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1, dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros(1, dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class SignedQuantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:
            self.register_buffer('quant_min_val', torch.tensor(-((1 << self.bits - 1) - 1), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor((1 << self.bits - 1) - 1, dtype=torch.float32))
        elif self.activation_weight_flag == 1:
            self.register_buffer('quant_min_val', torch.tensor(-(1 << self.bits - 1), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor((1 << self.bits - 1) - 1, dtype=torch.float32))
        else:
            None


class SymmetricQuantizer(SignedQuantizer):

    def update_qparams(self):
        self.q_type = 0
        quant_range = float(self.quant_max_val - self.quant_min_val) / 2
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))
        scale = float_range / quant_range
        scale = torch.max(scale, self.eps)
        zero_point = torch.zeros_like(scale)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


class QuantConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', a_bits=8, w_bits=8, q_type=0, q_level=0, weight_observer=0, quant_inference=False, qaft=False, ptq=False, percentile=0.9999):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            elif q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', a_bits=8, w_bits=8, q_type=0, weight_observer=0, quant_inference=False, qaft=False, ptq=False, percentile=0.9999):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, a_bits=8, w_bits=8, q_type=0, q_level=0, weight_observer=0, quant_inference=False, qaft=False, ptq=False, percentile=0.9999):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            elif q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


class QuantBNFuseConv2d(QuantConv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', eps=1e-05, momentum=0.1, a_bits=8, w_bits=8, q_type=0, q_level=0, weight_observer=0, pretrained_model=False, qaft=False, ptq=False, percentile=0.9999, bn_fuse_calib=False):
        super(QuantBNFuseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.num_flag = 0
        self.pretrained_model = pretrained_model
        self.qaft = qaft
        self.bn_fuse_calib = bn_fuse_calib
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(out_channels, dtype=torch.float32))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                elif q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            elif q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        if not self.qaft:
            if self.training:
                output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                dims = [dim for dim in range(4) if dim != 1]
                batch_mean = torch.mean(output, dim=dims)
                batch_var = torch.var(output, dim=dims)
                with torch.no_grad():
                    if not self.pretrained_model:
                        if self.num_flag == 0:
                            self.num_flag += 1
                            running_mean = batch_mean
                            running_var = batch_var
                        else:
                            running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                            running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                    else:
                        running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                        running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                if self.bias is not None:
                    bias_fused = reshape_to_bias(self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
                else:
                    bias_fused = reshape_to_bias(self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))
                if not self.bn_fuse_calib:
                    weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(batch_var + self.eps))
                else:
                    weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
            else:
                if self.bias is not None:
                    bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                else:
                    bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        else:
            if self.bias is not None:
                bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(weight_fused)
        if not self.qaft:
            if self.training:
                if not self.bn_fuse_calib:
                    output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation, self.groups)
                else:
                    output = F.conv2d(quant_input, quant_weight, None, self.stride, self.padding, self.dilation, self.groups)
                    output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
                    output += reshape_to_activation(bias_fused)
            else:
                output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation, self.groups)
        return output


class QuantReLU(nn.ReLU):

    def __init__(self, inplace=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantReLU, self).__init__(inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output


class QuantLeakyReLU(nn.LeakyReLU):

    def __init__(self, negative_slope=0.01, inplace=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantLeakyReLU, self).__init__(negative_slope, inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.leaky_relu(quant_input, self.negative_slope, self.inplace)
        return output


class QuantSigmoid(nn.Sigmoid):

    def __init__(self, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantSigmoid, self).__init__()
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.sigmoid(quant_input)
        return output


class QuantMaxPool2d(nn.MaxPool2d):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.max_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)
        return output


class QuantAvgPool2d(nn.AvgPool2d):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        return output


class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(quant_input, self.output_size)
        return output


class QuantAdd(nn.Module):

    def __init__(self, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAdd, self).__init__()
        if not ptq:
            self.observer_res = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            self.observer_shortcut = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
        else:
            self.observer_res = HistogramObserver(q_level='L', percentile=percentile)
            self.observer_shortcut = HistogramObserver(q_level='L', percentile=percentile)
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft, union=True)

    def forward(self, res, shortcut):
        self.observer_res(res)
        self.observer_shortcut(shortcut)
        observer_min_val = torch.min(self.observer_res.min_val, self.observer_shortcut.min_val)
        observer_max_val = torch.max(self.observer_res.max_val, self.observer_shortcut.max_val)
        self.activation_quantizer.observer.min_val = observer_min_val
        self.activation_quantizer.observer.max_val = observer_max_val
        quant_res = self.activation_quantizer(res)
        quant_shortcut = self.activation_quantizer(shortcut)
        output = quant_res + quant_shortcut
        return output


class SegmentationModule_v2_trt(nn.Module):

    def __init__(self, context, buffers, crit, deep_sup_scale=None, use_softmax=False, binding_id=0):
        super(SegmentationModule_v2_trt, self).__init__()
        self.context = context
        self.inputs, self.outputs, self.bindings, self.stream = buffers
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.use_softmax = use_softmax
        self.binding_id = binding_id

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, feed_dict, *, segSize=None, shape_of_input):
        shape_of_output = 1, 2, int(shape_of_input[2] / 8), int(shape_of_input[3] / 8)
        self.inputs[0].host = util_trt.to_numpy(feed_dict['img_data']).astype(dtype=np.float32).reshape(-1)
        trt_outputs, infer_time = util_trt.do_inference_v2(context=self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, h_=shape_of_input[2], w_=shape_of_input[3], binding_id=self.binding_id)
        trt_outputs = trt_outputs[0][:shape_of_output[0] * shape_of_output[1] * shape_of_output[2] * shape_of_output[3]]
        results = util_trt.postprocess_the_outputs(trt_outputs, shape_of_output)
        x = torch.from_numpy(results)
        x = x
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        pred = x
        return pred, infer_time


class C1_unet_v3(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1_unet_v3, self).__init__()
        self.use_softmax = use_softmax
        self.cbr1 = nn.Sequential(conv3x3_bn_relu(fc_dim, fc_dim, 1), conv3x3_bn_relu(fc_dim, fc_dim, 1), conv3x3_bn_relu(fc_dim, fc_dim, 1))
        self.cbr2 = nn.Sequential(conv3x3_bn_relu(fc_dim // 2, fc_dim // 2, 1), conv3x3_bn_relu(fc_dim // 2, fc_dim // 2, 1), conv3x3_bn_relu(fc_dim // 2, fc_dim // 2, 1))
        self.cbr3 = conv3x3_bn_relu(fc_dim // 2 * 3, fc_dim // 2 * 3, 1)
        self.cbr4 = nn.Sequential(conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1), conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1), conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1))
        self.cbr5 = nn.Sequential(conv3x3_bn_relu(fc_dim // 4 * 7, fc_dim // 2, 1), conv3x3_bn_relu(fc_dim // 2, fc_dim // 2, 1))
        self.conv_last = nn.Conv2d(fc_dim // 2, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        shape = list(conv_out[1].size())
        shape = shape[2:]
        shape2 = list(conv_out[2].size())
        shape2 = shape2[2:]
        x2 = self.cbr1(conv_out[3])
        x2 = nn.functional.interpolate(x2, size=(int(shape2[0]), int(shape2[1])), mode='bilinear', align_corners=False)
        x3 = self.cbr2(conv_out[2])
        x3 = torch.cat([x2, x3], 1)
        x3 = nn.functional.interpolate(x3, size=(int(shape[0]), int(shape[1])), mode='bilinear', align_corners=False)
        x3 = self.cbr3(x3)
        x4 = self.cbr4(conv_out[1])
        x4 = torch.cat([x4, x3], 1)
        x5 = self.cbr5(x4)
        x = self.conv_last(x5)
        """
        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        """
        return x


def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, channels, height, width)
    return x


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', eps=1e-05, momentum=0.1, channel_shuffle=0, shuffle_groups=1):
        super(ConvBNReLU, self).__init__()
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self, cfg=None):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]
        self.model = nn.Sequential(ConvBNReLU(3, cfg[0], kernel_size=5, stride=1, padding=2), ConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=0), ConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2), nn.MaxPool2d(kernel_size=2, stride=2, padding=0), ConvBNReLU(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2), ConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16), ConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4), nn.MaxPool2d(kernel_size=2, stride=2, padding=0), ConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4), ConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32), ConvBNReLU(cfg[7], 10, kernel_size=1, stride=1, padding=0), nn.AvgPool2d(kernel_size=8, stride=1, padding=0))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels * BasicBlock.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels * BasicBlock.expansion))
        self.add = Add()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.add(self.residual_function(x), self.shortcut(x)))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        self.add = Add()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.add(self.residual_function(x), self.shortcut(x)))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Add,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BottleNeck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBNReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HistogramObserver,
     lambda: ([], {'q_level': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (QuantAdaptiveAvgPool2d,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantAvgPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantBNFuseConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantMaxPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_666DZY666_micronet(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

