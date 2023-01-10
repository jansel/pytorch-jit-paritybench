import sys
_module = sys.modules[__name__]
del sys
enhancing = _module
dataloader = _module
cc3m = _module
classimage = _module
coco = _module
imagenet = _module
inatural = _module
lsun = _module
srimage = _module
textimage = _module
layers = _module
op = _module
conv2d_gradfix = _module
fused_act = _module
upfirdn2d = _module
segmentation = _module
vqperceptual = _module
clipcond = _module
dummycond = _module
vqcond = _module
layers = _module
quantizers = _module
vitvqgan = _module
layers = _module
transformer = _module
callback = _module
general = _module
scheduler = _module
tokenizer = _module
main = _module

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


from typing import Optional


from torch.utils.data import DataLoader


from typing import Union


from typing import Callable


from typing import Tuple


from typing import Any


from torchvision import transforms as T


from torch.utils.data import Dataset


import numpy as np


from random import randint


from random import choice


import torch


from torchvision.datasets import ImageFolder


from typing import List


from torchvision.datasets import ImageNet


from torchvision.datasets.utils import download_and_extract_archive


from torchvision.datasets.utils import verify_str_arg


from torchvision.datasets.vision import VisionDataset


from torchvision.datasets import LSUN


from torch import nn


from math import log2


from math import sqrt


from functools import partial


import torch.nn as nn


import torch.nn.functional as F


import warnings


from torch import autograd


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.cpp_extension import load


from collections import abc


from typing import Dict


import math


from collections import OrderedDict


from torch.optim import lr_scheduler


from torch.cuda.amp import autocast


from typing import Generic


import torchvision


import random


from typing import ClassVar


from functools import lru_cache


from typing import Set


class ActNorm(nn.Module):

    def __init__(self, num_features: int, logdet: Optional[bool]=False, affine: Optional[bool]=True, allow_reverse_init: Optional[bool]=False) ->None:
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: torch.FloatTensor) ->None:
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-06))

    def forward(self, input: torch.FloatTensor, reverse: Optional[bool]=False) ->Union[torch.FloatTensor, Tuple]:
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        _, _, height, width = input.shape
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        h = self.scale * (input + self.loc)
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0])
            return h, logdet
        return h

    def reverse(self, output: torch.FloatTensor) ->torch.FloatTensor:
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError('Initializing ActNorm in reverse direction is disabled by default. Use allow_reverse_init=True to enable.')
            else:
                self.initialize(output)
                self.initialized.fill_(1)
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        h = output / self.scale - self.loc
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


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
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


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
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None
        self.scale = 1 / sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        out = conv2d_gradfix.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
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
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = 1 / sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out


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


class StyleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / sqrt(2)
        return out


def weights_init(m: nn.Module) ->None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc: int=3, ndf: int=64, n_layers: int=3, use_actnorm: bool=False) ->None:
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def forward(self, input: torch.FloatTensor) ->torch.FloatTensor:
        """Standard forward."""
        return self.main(input)


class StyleDiscriminator(nn.Module):

    def __init__(self, size=256, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        log_size = int(log2(size))
        in_channel = channels[size]
        blocks = [ConvLayer(3, channels[size], 1)]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(StyleBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.blocks = nn.Sequential(*blocks)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))

    def forward(self, x):
        out = self.blocks(x)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        group = batch // (batch // group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
        return out.squeeze()


class BCELoss(nn.Module):

    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss, {}


class BCELossWithQuant(nn.Module):

    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target)
        loss = bce_loss + self.codebook_weight * qloss
        log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/bce_loss'.format(split): bce_loss.detach().mean(), '{}/quant_loss'.format(split): qloss.detach().mean()}
        return loss, log


class DummyLoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()


class VQLPIPS(nn.Module):

    def __init__(self, codebook_weight: float=1.0, loglaplace_weight: float=1.0, loggaussian_weight: float=1.0, perceptual_weight: float=1.0) ->None:
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net='vgg', verbose=False)
        self.codebook_weight = codebook_weight
        self.loglaplace_weight = loglaplace_weight
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor, optimizer_idx: int, global_step: int, batch_idx: int, last_layer: Optional[nn.Module]=None, split: Optional[str]='train') ->Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        loglaplace_loss = (reconstructions - inputs).abs().mean()
        loggaussian_loss = (reconstructions - inputs).pow(2).mean()
        perceptual_loss = self.perceptual_loss(inputs * 2 - 1, reconstructions * 2 - 1).mean()
        nll_loss = self.loglaplace_weight * loglaplace_loss + self.loggaussian_weight * loggaussian_loss + self.perceptual_weight * perceptual_loss
        loss = nll_loss + self.codebook_weight * codebook_loss
        log = {'{}/total_loss'.format(split): loss.clone().detach(), '{}/quant_loss'.format(split): codebook_loss.detach(), '{}/rec_loss'.format(split): nll_loss.detach(), '{}/loglaplace_loss'.format(split): loglaplace_loss.detach(), '{}/loggaussian_loss'.format(split): loggaussian_loss.detach(), '{}/perceptual_loss'.format(split): perceptual_loss.detach()}
        return loss, log


def hinge_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
    loss_fake = -logits_fake.mean() * 2 if logits_real is None else F.relu(1.0 + logits_fake).mean()
    loss_real = 0 if logits_real is None else F.relu(1.0 - logits_real).mean()
    return 0.5 * (loss_real + loss_fake)


def least_square_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
    loss_fake = logits_fake.pow(2).mean() * 2 if logits_real is None else (1 + logits_fake).pow(2).mean()
    loss_real = 0 if logits_real is None else (1 - logits_real).pow(2).mean()
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
    loss_fake = F.softplus(-logits_fake).mean() * 2 if logits_real is None else F.softplus(logits_fake).mean()
    loss_real = 0 if logits_real is None else F.softplus(-logits_real).mean()
    return 0.5 * (loss_real + loss_fake)


class DummyCond(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def encode(self, condition: Any) ->Tuple[Any, Any, Any]:
        return condition, None, condition

    def decode(self, condition: Any) ->Any:
        return condition

    def encode_codes(self, condition: Any) ->Any:
        return condition

    def decode_codes(self, condition: Any) ->Any:
        return condition


def get_obj_from_str(name: str, reload: bool=False) ->ClassVar:
    module, cls = name.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class ClassCond(DummyCond):

    def __init__(self, image_size: Union[Tuple[int, int], int], class_name: Union[str, List[str]]) ->None:
        super().__init__()
        self.img_size = image_size
        if isinstance(class_name, str):
            if class_name.endswith('txt') and os.path.isfile(class_name):
                self.cls_name = open(class_name, 'r').read().split('\n')
            elif '.' not in class_name and not os.path.isfile(class_name):
                self.cls_name = class_name
        elif isinstance(class_name, list) and isinstance(class_name[0], str):
            self.cls_name = class_name
        else:
            raise Exception('Class file format not supported')

    def to_img(self, clss: torch.LongTensor) ->torch.FloatTensor:
        W, H = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        font = ImageFont.truetype(os.path.join(os.getcwd(), 'assets', 'font', 'arial.ttf'), 12)
        imgs = []
        for cls in clss:
            cls_name = self.cls_name[int(cls)]
            length = 0
            img = Image.new('RGBA', (W, H), 'white')
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(cls_name, font)
            draw.text(((W - w) / 2, (H - h) / 2), cls_name, font=font, fill='black', align='center')
            img = img.convert('RGB')
            img = T.ToTensor()(img)
            imgs.append(img)
        return torch.stack(imgs, dim=0)


class PreNorm(nn.Module):

    def __init__(self, dim: int, fn: nn.Module) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) ->torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int) ->None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.FloatTensor) ->torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim: int, heads: int=8, dim_head: int=64) ->None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) ->torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) ->None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)), PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) ->torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class ViTEncoder(nn.Module):

    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int], dim: int, depth: int, heads: int, mlp_dim: int, channels: int=3, dim_head: int=64) ->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        self.num_patches = image_height // patch_height * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size), Rearrange('b c h w -> b (h w) c'))
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) ->torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)
        return x


class ViTDecoder(nn.Module):

    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int], dim: int, depth: int, heads: int, mlp_dim: int, channels: int=3, dim_head: int=64) ->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        self.num_patches = image_height // patch_height * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=image_height // patch_height), nn.ConvTranspose2d(dim, channels, kernel_size=patch_size, stride=patch_size))
        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) ->torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(token)
        x = self.to_pixel(x)
        return x

    def get_last_layer(self) ->nn.Parameter:
        return self.to_pixel[-1].weight


class BaseQuantizer(nn.Module):

    def __init__(self, embed_dim: int, n_embed: int, straight_through: bool=True, use_norm: bool=True, use_residual: bool=False, num_quantizers: Optional[int]=None) ->None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.use_residual = use_residual
        self.num_quantizers = num_quantizers
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()

    def quantize(self, z: torch.FloatTensor) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass

    def forward(self, z: torch.FloatTensor) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()
            losses = []
            encoding_indices = []
            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)
                encoding_indices.append(indices)
                losses.append(loss)
            losses, encoding_indices = map(partial(torch.stack, dim=-1), (losses, encoding_indices))
            loss = losses.mean()
        if self.straight_through:
            z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):

    def __init__(self, embed_dim: int, n_embed: int, beta: float=0.25, use_norm: bool=True, use_residual: bool=False, num_quantizers: Optional[int]=None, **kwargs) ->None:
        super().__init__(embed_dim, n_embed, True, use_norm, use_residual, num_quantizers)
        self.beta = beta

    def quantize(self, z: torch.FloatTensor) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + torch.sum(embedding_norm ** 2, dim=1) - 2 * torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm) ** 2) + torch.mean((z_qnorm - z_norm.detach()) ** 2)
        return z_qnorm, loss, encoding_indices


class GumbelQuantizer(BaseQuantizer):

    def __init__(self, embed_dim: int, n_embed: int, temp_init: float=1.0, use_norm: bool=True, use_residual: bool=False, num_quantizers: Optional[int]=None, **kwargs) ->None:
        super().__init__(embed_dim, n_embed, False, use_norm, use_residual, num_quantizers)
        self.temperature = temp_init

    def quantize(self, z: torch.FloatTensor, temp: Optional[float]=None) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        hard = not self.training
        temp = self.temperature if temp is None else temp
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        logits = -torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) - torch.sum(embedding_norm ** 2, dim=1) + 2 * torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        logits = logits.view(*z.shape[:-1], -1)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=-1, hard=hard)
        z_qnorm = torch.matmul(soft_one_hot, embedding_norm)
        logits = F.log_softmax(logits, dim=-1)
        loss = torch.sum(logits.exp() * (logits + math.log(self.n_embed)), dim=-1).mean()
        encoding_indices = soft_one_hot.argmax(dim=-1)
        return z_qnorm, loss, encoding_indices


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, ctx_len: int, cond_len: int, embed_dim: int, n_heads: int, attn_bias: bool, use_mask: bool=True):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)
        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer('mask', torch.ones(ctx_len, ctx_len), persistent=False)
            self.mask = torch.tril(self.mask).view(1, ctx_len, ctx_len)
            self.mask[:, :cond_len, :cond_len] = 1
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ww = torch.zeros(1, 1, embed_dim)
            for i in range(embed_dim):
                ww[0, 0, i] = i / (embed_dim - 1)
        self.time_mix = nn.Parameter(ww)

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        x = x.transpose(0, 1).contiguous()
        k = self.key(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        q = self.query(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        v = self.value(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        if use_cache:
            present = torch.stack([k, v])
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        if use_cache and layer_past is not None:
            att = torch.bmm(q, k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)
        else:
            att = torch.bmm(q, k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            if self.use_mask:
                mask = self.mask if T == self.ctx_len else self.mask[:, :T, :T]
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)
        y = y.transpose(0, 1).contiguous().view(T, B, C)
        y = self.proj(y)
        if use_cache:
            return y.transpose(0, 1).contiguous(), present
        else:
            return y.transpose(0, 1).contiguous()


class FFN(nn.Module):

    def __init__(self, embed_dim, mlp_bias):
        super().__init__()
        self.p0 = nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias)
        self.p1 = nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias)

    def forward(self, x):
        x = self.p0(x)
        x = torch.square(torch.relu(x))
        x = self.p1(x)
        return x


class Block(nn.Module):

    def __init__(self, ctx_len: int, cond_len: int, embed_dim: int, n_heads: int, mlp_bias: bool, attn_bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len, cond_len=cond_len, embed_dim=embed_dim, n_heads=n_heads, attn_bias=attn_bias, use_mask=True)
        self.mlp = FFN(embed_dim=embed_dim, mlp_bias=mlp_bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def sample(self, x, layer_past=None):
        attn, present = self.attn(self.ln1(x), use_cache=True, layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):

    def __init__(self, vocab_cond_size: int, vocab_img_size: int, embed_dim: int, cond_num_tokens: int, img_num_tokens: int, n_heads: int, n_layers: int, mlp_bias: bool=True, attn_bias: bool=True) ->None:
        super().__init__()
        self.img_num_tokens = img_num_tokens
        self.vocab_cond_size = vocab_cond_size
        self.tok_emb_cond = nn.Embedding(vocab_cond_size, embed_dim)
        self.pos_emb_cond = nn.Parameter(torch.zeros(1, cond_num_tokens, embed_dim))
        self.tok_emb_code = nn.Embedding(vocab_img_size, embed_dim)
        self.pos_emb_code = nn.Parameter(torch.zeros(1, img_num_tokens, embed_dim))
        self.blocks = [Block(ctx_len=cond_num_tokens + img_num_tokens, cond_len=cond_num_tokens, embed_dim=embed_dim, n_heads=n_heads, mlp_bias=mlp_bias, attn_bias=attn_bias) for i in range(1, n_layers + 1)]
        self.blocks = nn.Sequential(*self.blocks)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_img_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) ->None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, codes: torch.LongTensor, conds: torch.LongTensor) ->torch.FloatTensor:
        codes = codes.view(codes.shape[0], -1)
        codes = self.tok_emb_code(codes)
        conds = self.tok_emb_cond(conds)
        codes = codes + self.pos_emb_code
        conds = conds + self.pos_emb_cond
        x = torch.cat([conds, codes], axis=1).contiguous()
        x = self.blocks(x)
        x = self.layer_norm(x)
        x = x[:, conds.shape[1] - 1:-1].contiguous()
        logits = self.head(x)
        return logits

    def sample(self, conds: torch.LongTensor, top_k: Optional[float]=None, top_p: Optional[float]=None, softmax_temperature: float=1.0, use_fp16: bool=True) ->Tuple[torch.FloatTensor, torch.LongTensor]:
        past = codes = logits = None
        for i in range(self.img_num_tokens):
            if codes is None:
                codes_ = None
                pos_code = None
            else:
                codes_ = codes.clone().detach()
                codes_ = codes_[:, -1:]
                pos_code = self.pos_emb_code[:, i - 1:i, :]
            logits_, presents = self.sample_step(codes_, conds, pos_code, use_fp16, past)
            logits_ = logits_
            logits_ = logits_ / softmax_temperature
            presents = torch.stack(presents).clone().detach()
            if past is None:
                past = [presents]
            else:
                past.append(presents)
            if top_k is not None:
                v, ix = torch.topk(logits_, top_k)
                logits_[logits_ < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits_, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_idx_remove_cond = cum_probs >= top_p
                sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
                sorted_idx_remove_cond[..., 0] = 0
                indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            idx = torch.multinomial(probs, num_samples=1).clone().detach()
            codes = idx if codes is None else torch.cat([codes, idx], axis=1)
            logits = logits_ if logits is None else torch.cat([logits, logits_], axis=1)
        del past
        return logits, codes

    def sample_step(self, codes: torch.LongTensor, conds: torch.LongTensor, pos_code: torch.LongTensor, use_fp16: bool=True, past: Optional[torch.FloatTensor]=None) ->Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        with autocast(enabled=use_fp16):
            presents = []
            if codes is None:
                assert past is None
                conds = self.tok_emb_cond(conds)
                x = conds + self.pos_emb_cond
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.layer_norm(x)
                x = x[:, conds.shape[1] - 1].contiguous()
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes + pos_code
                past = torch.cat(past, dim=-2)
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)
                x = self.layer_norm(x)
                x = x[:, -1].contiguous()
            logits = self.head(x)
            return logits, presents


class RQTransformer(nn.Module):

    def __init__(self, vocab_cond_size: int, vocab_img_size: int, embed_dim: int, cond_num_tokens: int, img_num_tokens: int, depth_num_tokens: int, spatial_n_heads: int, depth_n_heads: int, spatial_n_layers: int, depth_n_layers: int, mlp_bias: bool=True, attn_bias: bool=True) ->None:
        super().__init__()
        self.img_num_tokens = img_num_tokens
        self.depth_num_tokens = depth_num_tokens
        self.vocab_img_size = vocab_img_size
        self.tok_emb_cond = nn.Embedding(vocab_cond_size, embed_dim)
        self.pos_emb_cond = nn.Parameter(torch.rand(1, cond_num_tokens, embed_dim))
        self.tok_emb_code = nn.Embedding(vocab_img_size, embed_dim)
        self.pos_emb_code = nn.Parameter(torch.rand(1, img_num_tokens, embed_dim))
        self.pos_emb_depth = nn.Parameter(torch.rand(1, depth_num_tokens - 1, embed_dim))
        self.spatial_transformer = [Block(ctx_len=cond_num_tokens + img_num_tokens, cond_len=cond_num_tokens, embed_dim=embed_dim, n_heads=spatial_n_heads, mlp_bias=mlp_bias, attn_bias=attn_bias) for i in range(1, spatial_n_layers + 1)]
        self.spatial_transformer = nn.Sequential(*self.spatial_transformer)
        self.depth_transformer = [Block(ctx_len=depth_num_tokens, cond_len=0, embed_dim=embed_dim, n_heads=depth_n_heads, mlp_bias=mlp_bias, attn_bias=attn_bias) for i in range(1, depth_n_layers + 1)]
        self.depth_transformer = nn.Sequential(*self.depth_transformer)
        self.ln_spatial = nn.LayerNorm(embed_dim)
        self.ln_depth = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_img_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) ->None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, codes: torch.LongTensor, conds: torch.LongTensor) ->torch.FloatTensor:
        codes = codes.view(codes.shape[0], -1, codes.shape[-1])
        codes = self.tok_emb_code(codes)
        conds = self.tok_emb_cond(conds)
        codes_cumsum = codes.cumsum(-1)
        codes_sum = codes_cumsum[..., -1, :]
        codes = codes_sum + self.pos_emb_code
        conds = conds + self.pos_emb_cond
        h = torch.cat([conds, codes], axis=1).contiguous()
        h = self.ln_spatial(self.spatial_transformer(h))
        h = h[:, conds.shape[1] - 1:-1].contiguous()
        v = codes_cumsum[..., :-1, :] + self.pos_emb_depth
        v = torch.cat([h.unsqueeze(2), v], axis=2).contiguous()
        v = v.view(-1, *v.shape[2:])
        v = self.depth_transformer(v)
        logits = self.head(self.ln_depth(v))
        return logits

    def sample(self, conds: torch.LongTensor, top_k: Optional[float]=None, top_p: Optional[float]=None, softmax_temperature: float=1.0, use_fp16: bool=True) ->Tuple[torch.FloatTensor, torch.LongTensor]:
        past = codes = logits = None
        B, T, D, S = conds.shape[0], self.img_num_tokens, self.depth_num_tokens, self.vocab_img_size
        for i in range(self.img_num_tokens):
            depth_past = None
            if codes is None:
                codes_ = None
                pos_code = None
            else:
                codes_ = codes.clone().detach()
                codes_ = codes_[:, -self.depth_num_tokens:]
                pos_code = self.pos_emb_code[:, i - 1:i, :]
            hidden, presents = self.sample_spatial_step(codes_, conds, pos_code, use_fp16, past)
            presents = torch.stack(presents).clone().detach()
            if past is None:
                past = [presents]
            else:
                past.append(presents)
            last_len = 0 if codes is None else codes.shape[-1]
            for d in range(self.depth_num_tokens):
                if depth_past is None:
                    codes_ = None
                    pos_depth = None
                else:
                    codes_ = codes.clone().detach()
                    codes_ = codes_[:, last_len:]
                    pos_depth = self.pos_emb_depth[:, d - 1:d, :]
                logits_, depth_presents = self.sample_depth_step(codes_, hidden, pos_depth, use_fp16, depth_past)
                logits_ = logits_
                logits_ = logits_ / softmax_temperature
                depth_presents = torch.stack(depth_presents).clone().detach()
                if depth_past is None:
                    depth_past = [depth_presents]
                else:
                    depth_past.append(depth_presents)
                if top_k is not None:
                    v, ix = torch.topk(logits_, top_k)
                    logits_[logits_ < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_, dim=-1)
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_idx_remove_cond = cum_probs >= top_p
                    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
                    sorted_idx_remove_cond[..., 0] = 0
                    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                idx = torch.multinomial(probs, num_samples=1).clone().detach()
                codes = idx if codes is None else torch.cat([codes, idx], axis=1)
                logits = logits_ if logits is None else torch.cat([logits, logits_], axis=1)
            del depth_past
        del past
        codes = codes.view(B, T, D)
        logits = logits.view(B * T, D, S)
        return logits, codes

    def sample_spatial_step(self, codes: torch.LongTensor, conds: torch.LongTensor, pos_code: torch.LongTensor, use_fp16: bool=True, past: Optional[torch.Tensor]=None) ->Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        with autocast(enabled=use_fp16):
            presents = []
            if codes is None:
                assert past is None
                conds = self.tok_emb_cond(conds)
                x = conds + self.pos_emb_cond
                for i, block in enumerate(self.spatial_transformer):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.ln_spatial(x)
                x = x[:, conds.shape[1] - 1:conds.shape[1]].contiguous()
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes.sum(1, keepdim=True) + pos_code
                past = torch.cat(past, dim=-2)
                for i, block in enumerate(self.spatial_transformer):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)
                x = self.ln_spatial(x)
                x = x[:, -1:].contiguous()
            return x, presents

    def sample_depth_step(self, codes: torch.LongTensor, hidden: torch.FloatTensor, pos_depth: torch.LongTensor, use_fp16: bool=True, past: Optional[torch.Tensor]=None) ->Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        with autocast(enabled=use_fp16):
            presents = []
            if codes is None:
                assert past is None
                x = hidden
                for i, block in enumerate(self.depth_transformer):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.ln_depth(x)
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes.sum(1, keepdim=True) + pos_depth
                past = torch.cat(past, dim=-2)
                for i, block in enumerate(self.depth_transformer):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)
            x = self.ln_depth(x)
            x = x[:, -1].contiguous()
            logits = self.head(x)
            return logits, presents


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BCELossWithQuant,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'ctx_len': 4, 'cond_len': 4, 'embed_dim': 4, 'n_heads': 4, 'mlp_bias': 4, 'attn_bias': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FFN,
     lambda: ([], {'embed_dim': 4, 'mlp_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FusedLeakyReLU,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GumbelQuantizer,
     lambda: ([], {'embed_dim': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadSelfAttention,
     lambda: ([], {'ctx_len': 4, 'cond_len': 4, 'embed_dim': 4, 'n_heads': 4, 'attn_bias': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PatchDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VectorQuantizer,
     lambda: ([], {'embed_dim': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_thuanz123_enhancing_transformers(_paritybench_base):
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

