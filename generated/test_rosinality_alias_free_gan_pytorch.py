import sys
_module = sys.modules[__name__]
del sys
config = _module
generate = _module
model = _module
op = _module
prepare_data = _module
apply_factor = _module
calc_inception = _module
closed_form_factorization = _module
convert_weight = _module
dataset = _module
distributed = _module
fid = _module
generate = _module
inception = _module
lpips = _module
base_model = _module
dist_model = _module
networks_basic = _module
pretrained_networks = _module
model = _module
non_leaking = _module
conv2d_gradfix = _module
fused_act = _module
upfirdn2d = _module
ppl = _module
projector = _module
swagan = _module
train = _module
train = _module

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


from torchvision import utils


import numpy as np


import math


from torch import nn


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.models import inception_v3


from torchvision.models import Inception3


from torch.utils.data import Dataset


from torch import distributed as dist


from torch.utils.data.sampler import Sampler


from scipy import linalg


import torch.nn as nn


import torch.nn.functional as F


from torchvision import models


from torch.autograd import Variable


from collections import OrderedDict


import itertools


from scipy.ndimage import zoom


import functools


import torch.nn.init as init


from collections import namedtuple


from torchvision import models as tv


import random


from torch.autograd import Function


from torch import autograd


import warnings


from torch.utils.cpp_extension import load


from collections import abc


from torch import optim


from torch.utils import data


import torch.distributed as dist


class FourierFeature(nn.Module):

    def __init__(self, size, dim, sampling_rate, cutoff, eps=1e-08):
        super().__init__()
        freqs = torch.randn(dim, 2)
        radii = freqs.square().sum(1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= cutoff
        phases = torch.rand(dim) - 0.5
        self.dim = dim
        self.size = torch.as_tensor(size).expand(2).tolist()
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, batch_size, affine, transform=None):
        freqs = self.freqs.unsqueeze(0)
        phases = self.phases.unsqueeze(0)
        norm = torch.norm(affine[:, :2], dim=-1, keepdim=True)
        affine = affine / norm
        m_rot = torch.eye(3, device=affine.device).unsqueeze(0).repeat(batch_size, 1, 1)
        m_rot[:, 0, 0] = affine[:, 0]
        m_rot[:, 0, 1] = -affine[:, 1]
        m_rot[:, 1, 0] = affine[:, 1]
        m_rot[:, 1, 1] = affine[:, 0]
        m_tra = torch.eye(3, device=affine.device).unsqueeze(0).repeat(batch_size, 1, 1)
        m_tra[:, 0, 2] = -affine[:, 2]
        m_tra[:, 1, 2] = -affine[:, 3]
        if transform is not None:
            transform = m_rot @ m_tra @ transform
        else:
            transform = m_rot @ m_tra
        phases = phases + (freqs @ transform[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transform[:, :2, :2]
        amplitude = (1 - (freqs.norm(dim=2) - self.cutoff) / (self.sampling_rate / 2 - self.cutoff)).clamp(0, 1)
        theta = torch.eye(2, 3, device=affine.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grid = F.affine_grid(theta.unsqueeze(0), (1, 1, self.size[1], self.size[0]), align_corners=False)
        x = (grid.unsqueeze(3) @ freqs.permute(0, 2, 1).view(-1, 1, 1, 2, self.dim)).squeeze(3)
        x = x + phases.view(-1, 1, 1, self.dim)
        x = torch.sin(x * (math.pi * 2))
        x = x * amplitude.view(-1, 1, 1, self.dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, up, down, pad, g_pad, in_size, out_size, flip_filter, gain):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_op.upfirdn2d(grad_output, kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, flip_filter, gain)
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
        ctx.flip_filter = flip_filter
        ctx.gain = gain
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1, not ctx.flip_filter, ctx.gain)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad, flip_filter, gain):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel)
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
        ctx.flip_filter = flip_filter
        ctx.gain = gain
        out = upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1, flip_filter, gain)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(grad_output, kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size, not ctx.flip_filter, ctx.gain)
        return grad_input, None, None, None, None, None, None


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1, flip_filter=False, gain=1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = kernel * gain ** (kernel.ndim / 2)
    if not flip_filter:
        w = torch.flip(w, list(range(kernel.ndim)))
    if kernel.ndim == 2:
        kernel_h, kernel_w = w.shape
        out = F.conv2d(out, w.view(1, 1, kernel_h, kernel_w))
    else:
        kernel_h = kernel_w = w.shape[0]
        out = F.conv2d(out, w.view(1, 1, kernel_h, 1))
        out = F.conv2d(out, w.view(1, 1, 1, kernel_h))
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0), flip_filter=False, gain=1):
    if not isinstance(up, abc.Iterable):
        up = up, up
    if not isinstance(down, abc.Iterable):
        down = down, down
    if len(pad) == 2:
        pad = pad[0], pad[1], pad[0], pad[1]
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, *up, *down, *pad, flip_filter=flip_filter, gain=gain)
    else:
        gain = gain ** (kernel.ndim / 2)
        if kernel.ndim == 1:
            out = UpFirDn2d.apply(input, kernel.unsqueeze(0), (up[0], 1), (down[0], 1), (*pad[:2], 0, 0), flip_filter, gain)
            out = UpFirDn2d.apply(out, kernel.unsqueeze(1), (1, up[1]), (1, down[1]), (0, 0, *pad[2:]), flip_filter, gain)
        else:
            out = UpFirDn2d.apply(input, kernel, up, down, pad, flip_filter, gain)
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
        bias = self.bias
        if bias is not None:
            bias = self.bias * self.lr_mul
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(input, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


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


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], fused=True):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, upsample={self.upsample}, downsample={self.downsample})'

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)
            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-08).rsqrt()
            input = input * style.reshape(batch, in_channel, 1, 1)
            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(input, weight, padding=0, stride=2)
                out = self.blur(out)
            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)
            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)
            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)
            return out
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = conv2d_gradfix.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class AliasFreeActivation(nn.Module):

    def __init__(self, out_channel, negative_slope, upsample_filter, downsample_filter, upsample, downsample, padding, clamp=None):
        super().__init__()
        self.register_buffer('upsample_filter', upsample_filter)
        self.register_buffer('downsample_filter', downsample_filter)
        self.negative_slope = negative_slope
        self.upsample = upsample
        self.downsample = downsample
        self.padding = padding
        self.clamp = clamp

    def forward(self, input, bias):
        out = filtered_lrelu(input, bias, self.upsample_filter, self.downsample_filter, self.upsample, self.downsample, self.padding, self.negative_slope, self.clamp)
        return out


class AliasFreeConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample_filter, downsample_filter, upsample=1, demodulate=True, padding=None, to_rgb=False, ema=0.999):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, demodulate=demodulate, padding=kernel_size - 1)
        self.bias = nn.Parameter(torch.zeros(out_channel))
        self.to_rgb = to_rgb
        if to_rgb:
            self.style_gain = 1 / in_channel ** 0.5
        else:
            self.style_gain = None
            self.activation = AliasFreeActivation(out_channel, 0.2, upsample_filter, downsample_filter, upsample, 2, padding=padding)
        self.ema = ema
        self.register_buffer('ema_var', torch.tensor(1.0))

    def forward(self, input, style):
        if self.training:
            var = input.detach().square().mean()
            self.ema_var.mul_(self.ema).add_(var, alpha=1 - self.ema)
        out = self.conv(input, style, input_gain=torch.rsqrt(self.ema_var), style_gain=self.style_gain)
        if self.to_rgb:
            out = out + self.bias.view(1, -1, 1, 1)
        else:
            out = self.activation(out, self.bias)
        return out


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / 2 ** 0.5 * torch.ones(1, 2)
    haar_wav_h = 1 / 2 ** 0.5 * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]
    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class InverseHaarTransform(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        return ll + lh + hl + hh


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


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


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class StyledConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class HaarTransform(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        return torch.cat((ll, lh, hl, hh), 1)


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

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            self.iwt = InverseHaarTransform(3)
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform(3)
        self.conv = ModulatedConv2d(in_channel, 3 * 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3 * 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)
            out = out + skip
        return out


class Generator(nn.Module):

    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)
        self.channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_size = int(math.log(size, 2)) - 1
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channel = self.channels[4]
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.iwt = InverseHaarTransform(3)
        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_is_latent=False, noise=None, randomize_noise=True):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t
        if len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = self.iwt(skip)
        if return_latents:
            return image, latent
        else:
            return image, None


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0]):
        super(PerceptualLoss, self).__init__()
        None
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids)
        None
        None

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.model.forward(target, pred)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):
    in_H = in_tens.shape[2]
    scale_factor = 1.0 * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = util.dssim(1.0 * util.tensor2im(in0.data), 1.0 * util.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var
        return ret_var


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


class Downsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


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


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out


class FromRGB(nn.Module):

    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)
        self.conv = ConvLayer(3 * 4, out_channel, 1)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)
        out = self.conv(input)
        if skip is not None:
            out = out + skip
        return input, out


class Discriminator(nn.Module):

    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()
        log_size = int(math.log(size, 2)) - 1
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.from_rgbs.append(FromRGB(channels[4]))
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))

    def forward(self, input):
        input = self.dwt(input)
        out = None
        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)
        _, out = self.from_rgbs[-1](input, out)
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConstantInput,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     False),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FusedLeakyReLU,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HaarTransform,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InverseHaarTransform,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_rosinality_alias_free_gan_pytorch(_paritybench_base):
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

