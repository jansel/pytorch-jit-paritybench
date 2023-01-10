import sys
_module = sys.modules[__name__]
del sys
appearance_control = _module
config = _module
data = _module
demo_appearance_dataset = _module
demo_dataset = _module
fashion_base_function = _module
fashion_data = _module
demo = _module
base_function = _module
base_module = _module
discriminator = _module
extraction_distribution_model = _module
inference = _module
attn_recon = _module
gan = _module
perceptual = _module
op = _module
conv2d_gradfix = _module
fused_act = _module
upfirdn2d = _module
predict = _module
mmfashion = _module
fashion_inference = _module
mask_rcnn_r50_fpn_1x = _module
prepare_data = _module
train = _module
trainers = _module
base = _module
extraction_distribution_trainer = _module
cudnn = _module
distributed = _module
face_crop = _module
io = _module
logging = _module
meters = _module
misc = _module
trainer = _module
visualization = _module
common = _module
linear_attention = _module

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


import collections


import numpy as np


import torch


import torch.optim as optim


import torch.nn.functional as F


import torch.utils.data


import math


import torchvision.transforms.functional as F


from torch.utils.data import Dataset


from torch import nn


from torch.nn import functional as F


import functools


import torch.nn as nn


import torchvision


import warnings


from torch import autograd


from torch.autograd import Function


from torch.utils.cpp_extension import load


from collections import abc


import random


import time


import torch.backends.cudnn as cudnn


import torch.distributed as dist


from torch.utils.tensorboard import SummaryWriter


from torch.utils.tensorboard.summary import hparams


from torch._six import string_classes


from torch.optim import Adam


from torch.optim import lr_scheduler


conv2d_gradfix_cache = dict()


def ensure_tuple(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    return xs


weight_gradients_disabled = False


def conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = ensure_tuple(stride, ndim)
    padding = ensure_tuple(padding, ndim)
    output_padding = ensure_tuple(output_padding, ndim)
    dilation = ensure_tuple(dilation, ndim)
    key = transpose, weight_shape, stride, padding, output_padding, dilation, groups
    if key in conv2d_gradfix_cache:
        return conv2d_gradfix_cache[key]
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [(input_shape[i + 2] - (output_shape[i + 2] - 1) * stride[i] - (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1)) for i in range(ndim)]


    class Conv2d(autograd.Function):

        @staticmethod
        def forward(ctx, input, weight, bias):
            if not transpose:
                out = F.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)
            else:
                out = F.conv_transpose2d(input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs)
            ctx.save_for_backward(input, weight)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = None, None, None
            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad_input = conv2d_gradfix(transpose=not transpose, weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, weight, None)
            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))
            return grad_input, grad_weight, grad_bias


    class Conv2dGradWeight(autograd.Function):

        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation('aten::cudnn_convolution_backward_weight' if not transpose else 'aten::cudnn_convolution_transpose_backward_weight')
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32]
            grad_weight = op(weight_shape, grad_output, input, padding, stride, dilation, groups, *flags)
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_grad_output, grad_grad_input = None, None
            if ctx.needs_input_grad[0]:
                grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)
            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad_grad_input = conv2d_gradfix(transpose=not transpose, weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, grad_grad_weight, None)
            return grad_grad_output, grad_grad_input
    conv2d_gradfix_cache[key] = Conv2d
    return Conv2d


class EqualConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class ExtractionOperation(nn.Module):

    def __init__(self, in_channel, num_label, match_kernel):
        super(ExtractionOperation, self).__init__()
        self.value_conv = EqualConv2d(in_channel, in_channel, match_kernel, 1, match_kernel // 2, bias=True)
        self.semantic_extraction_filter = EqualConv2d(in_channel, num_label, match_kernel, 1, match_kernel // 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.num_label = num_label

    def forward(self, value, recoder):
        key = value
        b, c, h, w = value.shape
        key = self.semantic_extraction_filter(self.feature_norm(key))
        extraction_softmax = self.softmax(key.view(b, -1, h * w))
        values_flatten = self.value_conv(value).view(b, -1, h * w)
        neural_textures = torch.einsum('bkm,bvm->bvk', extraction_softmax, values_flatten)
        recoder['extraction_softmax'].insert(0, extraction_softmax)
        recoder['neural_textures'].insert(0, neural_textures)
        return neural_textures, extraction_softmax

    def feature_norm(self, input_tensor):
        input_tensor = input_tensor - input_tensor.mean(dim=1, keepdim=True)
        norm = torch.norm(input_tensor, 2, 1, keepdim=True) + sys.float_info.epsilon
        out = torch.div(input_tensor, norm)
        return out


class DistributionOperation(nn.Module):

    def __init__(self, num_label, input_dim, match_kernel=3):
        super(DistributionOperation, self).__init__()
        self.semantic_distribution_filter = EqualConv2d(input_dim, num_label, kernel_size=match_kernel, stride=1, padding=match_kernel // 2)
        self.num_label = num_label

    def forward(self, query, extracted_feature, recoder):
        b, c, h, w = query.shape
        query = self.semantic_distribution_filter(query)
        query_flatten = query.view(b, self.num_label, -1)
        query_softmax = F.softmax(query_flatten, 1)
        values_q = torch.einsum('bkm,bkv->bvm', query_softmax, extracted_feature.permute(0, 2, 1))
        attn_out = values_q.view(b, -1, h, w)
        recoder['semantic_distribution'].append(query)
        return attn_out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_op.upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = up, up
    if not isinstance(down, abc.Iterable):
        down = down, down
    if len(pad) == 2:
        pad = pad[0], pad[1], pad[0], pad[1]
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)
    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)
    return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        if bias:
            grad_bias = grad_input.sum(dim).detach()
        else:
            grad_bias = empty
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input.contiguous(), gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        ctx.bias = bias is not None
        if bias is None:
            bias = empty
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale)
        if not ctx.bias:
            grad_bias = None
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == 'cpu':
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2) * scale
        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale
    else:
        return FusedLeakyReLUFunction.apply(input.contiguous(), bias, negative_slope, scale)


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class EncoderLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, use_extraction=False, num_label=None, match_kernel=None, num_extractions=2):
        super().__init__()
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
            stride = 2
            padding = 0
        else:
            self.blur = None
            stride = 1
            padding = kernel_size // 2
        self.conv = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias and not activate)
        self.activate = FusedLeakyReLU(out_channel, bias=bias) if activate else None
        self.use_extraction = use_extraction
        if self.use_extraction:
            self.extraction_operations = nn.ModuleList()
            for _ in range(num_extractions):
                self.extraction_operations.append(ExtractionOperation(out_channel, num_label, match_kernel))

    def forward(self, input, recoder=None):
        out = self.blur(input) if self.blur is not None else input
        out = self.conv(out)
        out = self.activate(out) if self.activate is not None else out
        if self.use_extraction:
            for extraction_operation in self.extraction_operations:
                extraction_operation(out, recoder)
        return out


class EqualTransposeConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        weight = self.weight.transpose(0, 1)
        out = conv2d_gradfix.conv_transpose2d(input, weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class DecoderLayer(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, upsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, use_distribution=True, num_label=16, match_kernel=3):
        super().__init__()
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.conv = EqualTransposeConv2d(in_channel, out_channel, kernel_size, stride=2, padding=0, bias=bias and not activate)
        else:
            self.conv = EqualConv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size // 2, bias=bias and not activate)
            self.blur = None
        self.distribution_operation = DistributionOperation(num_label, out_channel, match_kernel=match_kernel) if use_distribution else None
        self.activate = FusedLeakyReLU(out_channel, bias=bias) if activate else None
        self.use_distribution = use_distribution

    def forward(self, input, neural_texture=None, recoder=None):
        out = self.conv(input)
        out = self.blur(out) if self.blur is not None else out
        if self.use_distribution and neural_texture is not None:
            out_attn = self.distribution_operation(out, neural_texture, recoder)
            out = (out + out_attn) / math.sqrt(2)
        out = self.activate(out.contiguous()) if self.activate is not None else out
        return out


class Upsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * factor ** 2
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class ToRGB(nn.Module):

    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = EqualConv2d(in_channel, 3, 3, stride=1, padding=1)

    def forward(self, input, skip=None):
        out = self.conv(input)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate))
        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))
        super().__init__(*layers)


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class Encoder(nn.Module):

    def __init__(self, size, input_dim, channels, num_labels=None, match_kernels=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.first = EncoderLayer(input_dim, channels[size], 1)
        self.convs = nn.ModuleList()
        log_size = int(math.log(size, 2))
        self.log_size = log_size
        in_channel = channels[size]
        for i in range(log_size - 1, 3, -1):
            out_channel = channels[2 ** i]
            num_label = num_labels[2 ** i] if num_labels is not None else None
            match_kernel = match_kernels[2 ** i] if match_kernels is not None else None
            use_extraction = num_label and match_kernel
            conv = EncoderLayer(in_channel, out_channel, kernel_size=3, downsample=True, blur_kernel=blur_kernel, use_extraction=use_extraction, num_label=num_label, match_kernel=match_kernel)
            self.convs.append(conv)
            in_channel = out_channel

    def forward(self, input, recoder=None):
        out = self.first(input)
        for layer in self.convs:
            out = layer(out, recoder)
        return out


class Decoder(nn.Module):

    def __init__(self, size, channels, num_labels, match_kernels, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channel = channels[16]
        self.log_size = int(math.log(size, 2))
        for i in range(4, self.log_size + 1):
            out_channel = channels[2 ** i]
            num_label, match_kernel = num_labels[2 ** i], match_kernels[2 ** i]
            use_distribution = num_label and match_kernel
            upsample = i != 4
            base_layer = functools.partial(DecoderLayer, out_channel=out_channel, kernel_size=3, blur_kernel=blur_kernel, use_distribution=use_distribution, num_label=num_label, match_kernel=match_kernel)
            up = nn.Module()
            up.conv0 = base_layer(in_channel=in_channel, upsample=upsample)
            up.conv1 = base_layer(in_channel=out_channel, upsample=False)
            up.to_rgb = ToRGB(out_channel, upsample=upsample)
            self.convs.append(up)
            in_channel = out_channel
        self.num_labels, self.match_kernels = num_labels, match_kernels

    def forward(self, input, neural_textures, recoder):
        counter = 0
        out, skip = input, None
        for i, up in enumerate(self.convs):
            if self.num_labels[2 ** (i + 4)] and self.match_kernels[2 ** (i + 4)]:
                neural_texture_conv0 = neural_textures[counter]
                neural_texture_conv1 = neural_textures[counter + 1]
                counter += 2
            else:
                neural_texture_conv0, neural_texture_conv1 = None, None
            out = up.conv0(out, neural_texture=neural_texture_conv0, recoder=recoder)
            out = up.conv1(out, neural_texture=neural_texture_conv1, recoder=recoder)
            skip = up.to_rgb(out, skip)
        image = skip
        return image


class Discriminator(nn.Module):

    def __init__(self, size, channels, input_nc=3, blur_kernel=[1, 3, 3, 1], is_square_image=True):
        super().__init__()
        convs = [ConvLayer(input_nc, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        if is_square_image:
            self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))
        else:
            self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 2, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))

    def forward(self, input):
        out = self.convs(input)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out


class Generator(nn.Module):

    def __init__(self, size, semantic_dim, channels, num_labels, match_kernels, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.size = size
        self.reference_encoder = Encoder(size, 3, channels, num_labels, match_kernels, blur_kernel)
        self.skeleton_encoder = Encoder(size, semantic_dim, channels)
        self.target_image_renderer = Decoder(size, channels, num_labels, match_kernels, blur_kernel)

    def _cal_temp(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self, source_image, skeleton):
        output_dict = {}
        recoder = collections.defaultdict(list)
        skeleton_feature = self.skeleton_encoder(skeleton)
        _ = self.reference_encoder(source_image, recoder)
        neural_textures = recoder['neural_textures']
        output_dict['fake_image'] = self.target_image_renderer(skeleton_feature, neural_textures, recoder)
        output_dict['info'] = recoder
        return output_dict


class AttnReconLoss(nn.Module):

    def __init__(self, weights={(8): 1, (16): 0.5, (32): 0.25, (64): 0.125, (128): 0.0625}):
        super(AttnReconLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.weights = weights

    def forward(self, attn_dict, input_image, gt_image):
        softmax, query = attn_dict['extraction_softmax'], attn_dict['semantic_distribution']
        if isinstance(softmax, list) or isinstance(query, list):
            loss, weights = 0, 0
            for item_softmax, item_query in zip(softmax, query):
                h, w = item_query.shape[2:]
                gt_ = F.interpolate(gt_image, (h, w)).detach()
                input_ = F.interpolate(input_image, (h, w)).detach()
                estimated_target = self.cal_attn_image(input_, item_softmax, item_query)
                loss += self.l1loss(estimated_target, gt_) * self.weights[h]
                weights += self.weights[h]
            loss = loss / weights
        else:
            h, w = query.shape[2:]
            gt_ = F.interpolate(gt_image, (h, w))
            input_ = F.interpolate(input_image, (h, w))
            estimated_target = self.cal_attn_image(input_, softmax, query)
            loss = self.l1loss(estimated_target, gt_)
        return loss

    def cal_attn_image(self, input_image, softmax, query):
        b, num_label, h, w = query.shape
        if b != input_image.shape[0]:
            ib, ic, ih, iw = input_image.shape
            num_load_img = b // ib
            input_image = input_image[:, None].expand(ib, num_load_img, ic, ih, iw).contiguous()
        input_image = input_image.view(b, -1, h * w)
        extracted = torch.einsum('bkm,bvm->bvk', softmax, input_image)
        query = F.softmax(query.view(b, num_label, -1), 1)
        estimated_target = torch.einsum('bkm,bvk->bvm', query, extracted)
        estimated_target = estimated_target.view(b, -1, h, w)
        return estimated_target


@torch.jit.script
def fuse_math_min_mean_neg(x):
    """Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_pos(x):
    """Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class GANLoss(nn.Module):
    """GAN loss constructor.

    Args:
        gan_mode (str): Type of GAN loss. ``'hinge'``, ``'least_square'``,
            ``'non_saturated'``, ``'wasserstein'``.
        target_real_label (float): The desired output label for real images.
        target_fake_label (float): The desired output label for fake images.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode
        None

    def forward(self, dis_output, t_real, dis_update=True):
        """GAN loss computation.

        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(dis_output, list):
            loss = 0
            for dis_output_i in dis_output:
                assert isinstance(dis_output_i, torch.Tensor)
                loss += self.loss(dis_output_i, t_real, dis_update)
            return loss / len(dis_output)
        else:
            return self.loss(dis_output, t_real, dis_update)

    def loss(self, dis_output, t_real, dis_update=True):
        """GAN loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if not dis_update:
            assert t_real, 'The target should be real when updating the generator.'
        if self.gan_mode == 'non_saturated':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = F.binary_cross_entropy_with_logits(dis_output, target_tensor)
        elif self.gan_mode == 'least_square':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = 0.5 * F.mse_loss(dis_output, target_tensor)
        elif self.gan_mode == 'hinge':
            if dis_update:
                if t_real:
                    loss = fuse_math_min_mean_pos(dis_output)
                else:
                    loss = fuse_math_min_mean_neg(dis_output)
            else:
                loss = -torch.mean(dis_output)
        elif self.gan_mode == 'wasserstein':
            if t_real:
                loss = -torch.mean(dis_output)
            else:
                loss = torch.mean(dis_output)
        elif self.gan_mode == 'style_gan2':
            if t_real:
                loss = F.softplus(-dis_output).mean()
            else:
                loss = F.softplus(dis_output).mean()
        else:
            raise ValueError('Unexpected gan_mode {}'.format(self.gan_mode))
        return loss

    def get_target_tensor(self, dis_output, t_real):
        """Return the target vector for the binary cross entropy loss
        computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
        Returns:
            target (tensor): Target tensor vector.
        """
        if t_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = dis_output.new_tensor(self.real_label)
            return self.real_label_tensor.expand_as(dis_output)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = dis_output.new_tensor(self.fake_label)
            return self.fake_label_tensor.expand_as(dis_output)


class _PerceptualNetwork(nn.Module):
    """The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), 'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _vgg16(layers):
    """Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1', (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15): 'relu_3_3', (18): 'relu_4_1', (20): 'relu_4_2', (22): 'relu_4_3', (25): 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg19(layers):
    """Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1', (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15): 'relu_3_3', (17): 'relu_3_4', (20): 'relu_4_1', (22): 'relu_4_2', (24): 'relu_4_3', (26): 'relu_4_4', (29): 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def apply_imagenet_normalization(input):
    normalized_input = (input + 1) / 2
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


class PerceptualLoss(nn.Module):

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None, criterion='l1', resize=False, resize_mode='bilinear', instance_normalized=False, num_scales=1, use_style_loss=False, weight_style_to_perceptual=0):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.0] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]
        assert len(layers) == len(weights), 'The number of layers (%s) must be equal to the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)
        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.use_style_loss = use_style_loss
        self.weight_style = weight_style_to_perceptual
        None
        None

    def forward(self, inp, target, mask=None):
        """Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        self.model.eval()
        inp, target = apply_imagenet_normalization(inp), apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(inp, mode=self.resize_mode, size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode=self.resize_mode, size=(224, 224), align_corners=False)
        loss = 0
        style_loss = 0
        for scale in range(self.num_scales):
            input_features, target_features = self.model(inp), self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)
                if mask is not None:
                    mask_ = F.interpolate(mask, input_feature.shape[2:], mode='bilinear', align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                loss += weight * self.criterion(input_feature, target_feature)
                if self.use_style_loss and scale == 0:
                    style_loss += self.criterion(self.compute_gram(input_feature), self.compute_gram(target_feature))
            if scale != self.num_scales - 1:
                inp = F.interpolate(inp, mode=self.resize_mode, scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(target, mode=self.resize_mode, scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
        if self.use_style_loss:
            return loss + style_loss * self.weight_style
        else:
            return loss

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G


class LoadImage(object):

    def __call__(self, results):
        if 'filename' not in results:
            results['filename'] = None
        results['img'] = results['img'][:, :, ::-1]
        results['img_shape'] = results['img'].shape
        results['ori_shape'] = results['img'].shape
        return results


class FashionInference(nn.Module):

    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        super(FashionInference, self).__init__()
        self.seg_model = init_detector(config_path, checkpoint_path, device=device)
        self.cfg = self.seg_model.cfg
        self.test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(self.test_pipeline)
        self.device = device
        self.classes = self.seg_model.CLASSES

    def forward(self, image_array, find_items, filename=None, score_thr=0.3):
        data = dict(img=image_array, filename=filename)
        data = self.test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
        with torch.no_grad():
            result = self.seg_model(return_loss=False, rescale=True, **data)
        bbox_result, segm_result = result
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        mask = 0
        for i in inds:
            i = int(i)
            label = self.classes[labels[i]]
            if label in self.mapping_classes(find_items):
                mask += maskUtils.decode(segms[i])
        return torch.tensor(mask > 0).float()[None, None]

    def mapping_classes(self, find_items):
        return_items = []
        for item in find_items:
            if item == 'up':
                return_items.extend(['top', 'dress', 'outer'])
            elif item == 'down':
                return_items.extend(['leggings', 'skirt', 'pants', 'belt', 'footwear'])
            else:
                return_items.append(item)
        return return_items


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'size': 4, 'channels': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'num_labels': 4, 'match_kernels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FusedLeakyReLU,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_RenYurui_Neural_Texture_Extraction_Distribution(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

