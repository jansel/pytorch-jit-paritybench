import sys
_module = sys.modules[__name__]
del sys
setup = _module
torchtools = _module
lr_scheduler = _module
delayed = _module
nn = _module
adain = _module
alias_free_activation = _module
equal_layers = _module
evonorm2d = _module
fourier_features = _module
functional = _module
gradient_penalty = _module
perceptual = _module
vq = _module
gp_loss = _module
mish = _module
modulation = _module
perceptual = _module
pixel_normalzation = _module
pos_embeddings = _module
simple_self_attention = _module
stylegan2 = _module
upfirdn2d = _module
transformers = _module
vq = _module
optim = _module
lamb = _module
lookahead = _module
novograd = _module
over9000 = _module
radam = _module
ralamb = _module
ranger = _module
utils = _module
diffusion = _module

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


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import CosineAnnealingLR


import torch


from torch import nn


import math


import torch.nn as nn


from torch import autograd


from torch.autograd import Function


import torch.nn.functional as F


from collections import abc


from torch.nn import functional as F


from torch.utils.cpp_extension import load


import collections


from torch.optim import Optimizer


from torch.optim.optimizer import Optimizer


from collections import defaultdict


import itertools as it


from torch.optim.optimizer import required


class AdaIN(nn.Module):

    def __init__(self, n_channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(n_channels)

    def forward(self, image, style):
        factor, bias = style.view(style.size(0), style.size(1), 1, 1).chunk(2, dim=1)
        result = self.norm(image) * factor + bias
        return result


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


class AliasFreeActivation(nn.Module):

    def __init__(self, activation, level, max_levels, max_size, max_channels, margin, start_cutoff=2, critical_layers=2, window_size=6):
        super().__init__()
        self.activation = activation
        self.cutoff, self.stopband, self.band_half, self.channels, self.size = self.alias_level_params(level, max_levels, max_size, max_channels, start_cutoff, critical_layers)
        self.cutoff_prev, self.stopband_prev, self.band_half_prev, self.channels_prev, self.size_prev = self.alias_level_params(max(level - 1, 0), max_levels, max_size, max_channels, start_cutoff, critical_layers)
        self.scale_factor = 2 if self.size_prev < self.size else 1
        up_filter = self._lowpass_filter(window_size * self.scale_factor * 2, self.cutoff_prev, self.band_half_prev, self.size * self.scale_factor * 2)
        self.register_buffer('up_filter', up_filter / up_filter.sum() * 2 * self.scale_factor)
        down_filter = self._lowpass_filter(window_size * self.scale_factor, self.cutoff, self.band_half, self.size * self.scale_factor * 2)
        self.register_buffer('down_filter', down_filter / down_filter.sum())
        p = self.up_filter.shape[0] - 2 * self.scale_factor
        self.up_pad = (p + 1) // 2 + 2 * self.scale_factor - 1, p // 2
        p = self.down_filter.shape[0] - 2
        self.down_pad = (p + 1) // 2, p // 2
        self.margin = margin

    @staticmethod
    def alias_level_params(level, max_levels, max_size, max_channels, start_cutoff=2, critical_layers=2, base_channels=2 ** 14):
        end_cutoff = max_size // 2
        cutoff = start_cutoff * (end_cutoff / start_cutoff) ** min(level / (max_levels - critical_layers), 1)
        start_stopband = start_cutoff ** 2.1
        end_stopband = end_cutoff * 2 ** 0.3
        stopband = start_stopband * (end_stopband / start_stopband) ** min(level / (max_levels - critical_layers), 1)
        size = 2 ** math.ceil(math.log(min(2 * stopband, max_size), 2))
        band_half = max(stopband, size / 2) - cutoff
        channels = min(round(base_channels / size), max_channels)
        return cutoff, stopband, band_half, channels, size

    def _lowpass_filter(self, n_taps, cutoff, band_half, sr):
        window = self._kaiser_window(n_taps, band_half, sr)
        ind = torch.arange(n_taps) - (n_taps - 1) / 2
        lowpass = 2 * cutoff / sr * torch.sinc(2 * cutoff / sr * ind) * window
        return lowpass

    def _kaiser_window(self, n_taps, f_h, sr):
        beta = self._kaiser_beta(n_taps, f_h, sr)
        ind = torch.arange(n_taps) - (n_taps - 1) / 2
        return torch.i0(beta * torch.sqrt(1 - (2 * ind / (n_taps - 1)) ** 2)) / torch.i0(torch.tensor(beta))

    def _kaiser_attenuation(self, n_taps, f_h, sr):
        df = 2 * f_h / (sr / 2)
        return 2.285 * (n_taps - 1) * math.pi * df + 7.95

    def _kaiser_beta(self, n_taps, f_h, sr):
        atten = self._kaiser_attenuation(n_taps, f_h, sr)
        if atten > 50:
            return 0.1102 * (atten - 8.7)
        elif 50 >= atten >= 21:
            return 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)
        else:
            return 0.0

    def forward(self, x):
        x = self._upsample(x, self.up_filter, 2 * self.scale_factor, pad=self.up_pad)
        x = self.activation(x)
        x = self._downsample(x, self.down_filter, 2, pad=self.down_pad)
        if self.scale_factor > 1 and self.margin > 0:
            m = self.scale_factor * self.margin // 2
            x = x[:, :, m:-m, m:-m]
        return x

    def _upsample(self, x, kernel, factor, pad=(0, 0)):
        x = upfirdn2d(x, kernel.unsqueeze(0), up=(factor, 1), pad=(*pad, 0, 0))
        x = upfirdn2d(x, kernel.unsqueeze(1), up=(1, factor), pad=(0, 0, *pad))
        return x

    def _downsample(self, x, kernel, factor, pad=(0, 0)):
        x = upfirdn2d(x, kernel.unsqueeze(0), down=(factor, 1), pad=(*pad, 0, 0))
        x = upfirdn2d(x, kernel.unsqueeze(1), down=(1, factor), pad=(0, 0, *pad))
        return x

    def extra_repr(self):
        info_string = f'cutoff={self.cutoff}, stopband={self.stopband}, band_half={self.band_half}, channels={self.channels}, size={self.size}'
        return info_string


class EqualLinear(nn.Linear):

    def __init__(self, *args, bias_init=0, lr_mul=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = 1 / math.sqrt(self.in_features) * lr_mul
        self.lr_mul = lr_mul
        nn.init.normal_(self.weight, std=1 / lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, bias_init)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class EqualConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fan_in = self.in_channels * self.kernel_size[0] ** 2
        self.scale = 1 / math.sqrt(fan_in)
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.scale, self.bias)


class EqualLeakyReLU(nn.LeakyReLU):

    def __init__(self, *args, scale=2 ** 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def forward(self, x):
        return super().forward(x) * self.scale


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


def group_std(x, groups=32, eps=1e-05):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


def instance_std(x, eps=1e-05):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


class EvoNorm2D(nn.Module):

    def __init__(self, input, non_linear=True, version='S0', efficient=False, affine=True, momentum=0.9, eps=1e-05, groups=32, training=True):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == 'S0':
            self.swish = MemoryEfficientSwish()
        self.groups = groups
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError('Invalid EvoNorm version')
        self.insize = input
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(self.v * x)
                else:
                    num = self.swish(x)
                return num / group_std(x, groups=self.groups, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var
            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class FourierFeatures2d(nn.Module):

    def __init__(self, size, dim, cutoff, affine_eps=1e-08, freq_range=[-0.5, 0.5], w_scale=0, allow_scaling=False, op_order=['r', 't', 's']):
        super().__init__()
        self.size = size
        self.dim = dim
        self.cutoff = cutoff
        self.freq_range = freq_range
        self.affine_eps = affine_eps
        self.w_scale = w_scale
        coords = torch.linspace(freq_range[0], freq_range[1], size + 1)[:-1]
        freqs = torch.linspace(0, cutoff, dim // 4)
        if w_scale > 0:
            freqs = freqs @ (torch.randn(dim // 4, dim // 4) * w_scale)
        coord_map = torch.outer(freqs, coords)
        coord_map = 2 * math.pi * coord_map
        self.register_buffer('coord_h', coord_map.view(freqs.shape[0], 1, size))
        self.register_buffer('coord_w', self.coord_h.transpose(1, 2).detach())
        self.register_buffer('lf', freqs.view(1, dim // 4, 1, 1) * 2 * math.pi * 2 / size)
        self.allow_scaling = allow_scaling
        for op in op_order:
            assert op in ['r', 't', 's'], f'Operation not valid: {op}'
        self.op_order = op_order

    def forward(self, affine):
        norm = ((affine[:, 0:1].pow(2) + affine[:, 1:2].pow(2)).sqrt() + self.affine_eps).expand(affine.size(0), 4)
        if self.allow_scaling:
            assert affine.size(-1) == 6, f'If scaling is enabled, 2 extra values must be passed for a total of 6, and not {affine.size(-1)}'
            norm = torch.cat([norm, norm.new_ones(affine.size(0), 2)], dim=1)
        else:
            assert affine.size(-1) == 4, f'If scaling is disabled, 4 affine values should be passed, and not {affine.size(-1)}'
        affine = affine / norm
        affine = affine[:, :, None, None, None]
        coord_h, coord_w = self.coord_h.unsqueeze(0), self.coord_w.unsqueeze(0)
        for op in reversed(self.op_order):
            if op == 's' and self.allow_scaling:
                coord_h = coord_h / nn.functional.threshold(affine[:, 5], 1.0, 1.0)
                coord_w = coord_w / nn.functional.threshold(affine[:, 4], 1.0, 1.0)
            elif op == 't':
                coord_h = coord_h - affine[:, 3] * self.lf
                coord_w = coord_w - affine[:, 2] * self.lf
            elif op == 'r':
                _coord_h = -coord_w * affine[:, 1] + coord_h * affine[:, 0]
                coord_w = coord_w * affine[:, 0] + coord_h * affine[:, 1]
                coord_h = _coord_h
        coord_h = torch.cat((torch.sin(coord_h), torch.cos(coord_h)), 1)
        coord_w = torch.cat((torch.sin(coord_w), torch.cos(coord_w)), 1)
        coords = torch.cat((coord_h, coord_w), 1)
        return coords

    def extra_repr(self):
        info_string = f'size={self.size}, dim={self.dim}, cutoff={self.cutoff}, freq_range={self.freq_range}'
        if self.w_scale > 0:
            info_string += f', w_scale={self.w_scale}'
        if self.allow_scaling:
            info_string += f', allow_scaling={self.allow_scaling}'
        return info_string


def gradient_penalty(netD, real_data, fake_data, l=10):
    batch_size = real_data.size(0)
    alpha = real_data.new_empty((batch_size, 1, 1, 1)).uniform_(0, 1)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=real_data.new_ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l
    return gradient_penalty


class GPLoss(nn.Module):

    def __init__(self, discriminator, l=10):
        super(GPLoss, self).__init__()
        self.discriminator = discriminator
        self.l = l

    def forward(self, real_data, fake_data):
        return gradient_penalty(self.discriminator, real_data, fake_data, self.l)


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ModulatedConv2d(nn.Conv2d):

    def __init__(self, *args, demodulate=True, ema_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        fan_in = self.in_channels * self.kernel_size[0] ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.demodulate = demodulate
        self.ema_decay = ema_decay
        self.register_buffer('ema_var', torch.tensor(1.0))
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, w):
        batch, in_channels, height, width = x.shape
        style = w.view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight.unsqueeze(0) * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        weight = weight.view(batch * self.out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        if self.ema_decay < 1:
            if self.training:
                var = x.pow(2).mean((0, 1, 2, 3))
                self.ema_var.mul_(self.ema_decay).add_(var.detach(), alpha=1 - self.ema_decay)
            weight = weight / (torch.sqrt(self.ema_var) + 1e-08)
        input = x.view(1, batch * in_channels, height, width)
        self.groups = batch
        out = self._conv_forward(input, weight, None)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


def total_variation(X, reduction='sum'):
    tv_h = torch.abs(X[:, :, :, 1:] - X[:, :, :, :-1])
    tv_v = torch.abs(X[:, :, 1:] - X[:, :, :-1])
    tv = torch.mean(tv_h) + torch.mean(tv_v) if reduction == 'mean' else torch.sum(tv_h) + torch.sum(tv_v)
    return tv


class TVLoss(nn.Module):

    def __init__(self, reduction='sum', alpha=0.0001):
        super(TVLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, x):
        return total_variation(x, reduction=self.reduction) * self.alpha


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
    """Create and initialize a `nn.Conv1d` layer with spectral normalization."""
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)


class SimpleSelfAttention(nn.Module):

    def __init__(self, n_in, ks=1, sym=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
        self.gamma = nn.Parameter(torch.Tensor([0.0]))
        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)
        size = x.size()
        x = x.view(*size[:2], -1)
        convx = self.conv(x)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class GPTTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.ln1(x)
        x = x + self.attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class vector_quantize(Function):

    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            x_sqr = torch.sum(x ** 2, dim=1, keepdim=True)
            dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)
            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)
            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)
        return grad_inputs, grad_codebook


class VectorQuantize(nn.Module):

    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
		Takes an input of variable size (as long as the last dimension matches the embedding size).
		Returns one tensor containing the nearest neigbour embeddings to each of the inputs, 
		with the same size as the input, vq and commitment components for the loss as a touple 
		in the second output and the indices of the quantized vectors in the third: 
		quantized, (vq_loss, commit_loss), indices
		"""
        super(VectorQuantize, self).__init__()
        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.vq = vector_quantize.apply
        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer('ema_element_count', torch.ones(k))
            self.register_buffer('ema_weight_sum', torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return (x + epsilon) / (n + x.size(0) * epsilon) * n

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)
        self.ema_element_count = self.ema_decay * self.ema_element_count + (1 - self.ema_decay) * elem_count
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-05)
        self.ema_weight_sum = self.ema_decay * self.ema_weight_sum + (1 - self.ema_decay) * weight_sum
        self.codebook.weight.data = self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()
        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


class binarize(Function):

    @staticmethod
    def forward(ctx, x, threshold=0.5):
        with torch.no_grad():
            binarized = (x > threshold).float()
            ctx.mark_non_differentiable(binarized)
            return binarized

    @staticmethod
    def backward(ctx, grad_output):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        return grad_inputs


class Binarize(nn.Module):

    def __init__(self, threshold=0.5):
        """
		Takes an input of any size.
		Returns an output of the same size but with its values binarized (0 if input is below a threshold, 1 if its above)
		"""
        super(Binarize, self).__init__()
        self.bin = binarize.apply
        self.threshold = threshold

    def forward(self, x):
        return self.bin(x, self.threshold)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaIN,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 1, 1])], {}),
     True),
    (AliasFreeActivation,
     lambda: ([], {'activation': _mock_layer(), 'level': 4, 'max_levels': 4, 'max_size': 4, 'max_channels': 4, 'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Binarize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EqualConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EqualLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FourierFeatures2d,
     lambda: ([], {'size': 4, 'dim': 4, 'cutoff': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GPTTransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModulatedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 4, 1, 1])], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleSelfAttention,
     lambda: ([], {'n_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VectorQuantize,
     lambda: ([], {'embedding_size': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_pabloppp_pytorch_tools(_paritybench_base):
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

