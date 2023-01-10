import sys
_module = sys.modules[__name__]
del sys
cuda_op = _module
fused_act = _module
op_native = _module
upfirdn2d = _module
demo = _module
metrics = _module
attribute_consistency = _module
eval_encoder = _module
fid = _module
ppl = _module
models = _module
anycost_gan = _module
dynamic_channel = _module
encoder = _module
ops = _module
LBFGS = _module
celeba_hq_split = _module
inception = _module
manipulator = _module
align_face = _module
calc_inception = _module
extract_edit_directions = _module
project = _module
train_gan = _module
utils = _module
datasets = _module
losses = _module
torch_utils = _module
train_utils = _module

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


from torch.autograd import Function


from torch.utils.cpp_extension import load


from torch.nn import functional as F


from torch.nn.functional import leaky_relu


import numpy as np


import time


import math


import torch.nn as nn


from torchvision import transforms


from scipy import linalg


from torchvision import models


import random


from torchvision.models import resnet50


import torch.nn.functional as F


from functools import reduce


from copy import deepcopy


from torch.optim import Optimizer


from torch.utils.data import DataLoader


from torch import optim


from torch.utils import data


from torchvision import utils


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


from torch import autograd


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return self.scale * leaky_relu(x + self.bias.reshape((1, -1, 1, 1))[:, :x.shape[1]], self.negative_slope, inplace=True)


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)
        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            assert self.first_k_oup <= out.shape[1]
            return out[:, :self.first_k_oup]
        else:
            return out


def fused_leaky_relu(input_, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * leaky_relu(input_ + bias[:input_.shape[1]], negative_slope, inplace=True)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            if self.activation == 'lrelu':
                out = fused_leaky_relu(out, self.bias * self.lr_mul)
            else:
                raise NotImplementedError
        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


G_CHANNEL_CONFIG = {(4): 4096, (8): 2048, (16): 1024, (32): 512, (64): 256, (128): 128, (256): 64, (512): 32, (1024): 16}


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    k = torch.flip(k, [0, 1])
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
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
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
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    else:
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        assert not downsample, 'Downsample is not implemented yet!'
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, upsample={self.upsample}, downsample={self.downsample})'

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style)
        style = style.view(batch, 1, -1, 1, 1)
        first_k_oup = self.first_k_oup if hasattr(self, 'first_k_oup') and self.first_k_oup is not None else self.weight.shape[1]
        assert first_k_oup <= self.weight.shape[1]
        weight = self.weight
        weight = weight[:, :first_k_oup, :in_channel].contiguous()
        weight = self.scale * weight * style[:, :, :in_channel]
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + self.eps)
        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.transpose(1, 2)
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])
            out = self.blur(out)
        else:
            x = x.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.view(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])
        return out


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

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=(1, 3, 3, 1), demodulate=True, activation='lrelu'):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        if activation == 'lrelu':
            self.activate = FusedLeakyReLU(out_channel)
        else:
            raise NotImplementedError

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
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

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class Generator(nn.Module):

    def __init__(self, resolution, style_dim=512, n_mlp=8, channel_multiplier=2, channel_max=512, blur_kernel=(1, 3, 3, 1), lr_mlp=0.01, act_func='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.style_dim = style_dim
        self.channel_max = channel_max
        style_mlp = [EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='lrelu') for _ in range(n_mlp)]
        style_mlp.insert(0, PixelNorm())
        self.style = nn.Sequential(*style_mlp)
        self.channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in G_CHANNEL_CONFIG.items()}
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, activation=act_func)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_res = int(math.log(resolution, 2))
        self.num_layers = (self.log_res - 2) * 2 + 1
        self.n_style = self.log_res * 2 - 2
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channel = self.channels[4]
        for i in range(3, self.log_res + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel, activation=act_func))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, activation=act_func))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

    def make_noise(self):
        device = self.style[-1].weight.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_res + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_style(self, n_sample):
        z = torch.randn(n_sample, self.style_dim, device=self.style[-1].weight.device)
        w = self.style(z).mean(0, keepdim=True)
        return w

    def get_style(self, z):
        z_shape = z.shape
        return self.style(z.view(-1, z.shape[-1])).view(z_shape)

    def forward(self, styles, return_styles=False, inject_index=None, truncation=1, truncation_style=None, input_is_style=False, noise=None, randomize_noise=True, return_rgbs=False, target_res=None):
        """
        :param styles: the input z or w, depending on input_is_style arg
        :param return_styles: whether to return w (used for training)
        :param inject_index: manually assign injection index
        :param truncation: whether to apply style truncation. default: no truncate
        :param truncation_style: the mean style used for truncation
        :param input_is_style: whether the input is style (w) or z
        :param noise: manually assign noise tensor per layer
        :param randomize_noise: whether to randomly draw the noise or use the fixed noise
        :param return_rgbs: whether to return all the lower resolution outputs
        :param target_res: assign target resolution; rarely used here
        :return: output image, _
        """
        assert len(styles.shape) == 3
        if not input_is_style:
            styles = self.get_style(styles)
        if truncation < 1:
            styles = (1 - truncation) * truncation_style.view(1, 1, -1) + truncation * styles
        if styles.shape[1] == 1:
            styles = styles.repeat(1, self.n_style, 1)
        elif styles.shape[1] == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_style - 1)
            style1 = styles[:, 0:1].repeat(1, inject_index, 1)
            style2 = styles[:, 1:2].repeat(1, self.n_style - inject_index, 1)
            styles = torch.cat([style1, style2], 1)
        else:
            assert styles.shape[1] == self.n_style
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]
        all_rgbs = []
        out = self.input(styles.shape[0])
        out = self.conv1(out, styles[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, styles[:, 1])
        all_rgbs.append(skip)
        if hasattr(self, 'target_res') and target_res is None:
            target_res = self.target_res
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, styles[:, i], noise=noise1)
            out = conv2(out, styles[:, i + 1], noise=noise2)
            skip = to_rgb(out, styles[:, i + 2], skip)
            all_rgbs.append(skip)
            i += 2
            if target_res is not None and skip.shape[-1] == target_res:
                break
        if return_styles:
            return skip, styles
        elif return_rgbs:
            return skip, all_rgbs
        else:
            return skip, None


class AdaptiveModulate(nn.Module):

    def __init__(self, num_features, g_arch_len):
        super(AdaptiveModulate, self).__init__()
        self.weight_mapping = nn.Linear(g_arch_len, num_features)
        self.bias_mapping = nn.Linear(g_arch_len, num_features)

    def forward(self, x, g_arch):
        assert x.dim() == 4
        weight = self.weight_mapping(g_arch.view(1, -1)).view(-1) + 1.0
        bias = self.bias_mapping(g_arch.view(1, -1)).view(-1)
        return x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)


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

    def forward(self, x):
        in_channel = x.shape[1]
        weight = self.weight
        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            weight = weight[:self.first_k_oup]
        weight = weight[:, :in_channel].contiguous()
        out = F.conv2d(x, weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=(1, 3, 3, 1), bias=True, activate='lrelu', modulate=False, g_arch_len=18 * 4):
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
        if modulate:
            layers.append(AdaptiveModulate(out_channel, g_arch_len))
        assert bias == (activate != 'none')
        if activate == 'lrelu':
            layers.append(FusedLeakyReLU(out_channel))
        else:
            assert activate == 'none'
        super().__init__(*layers)

    def forward(self, x, g_arch=None):
        for module in self:
            if isinstance(module, AdaptiveModulate):
                x = module(x, g_arch)
            else:
                x = module(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=(1, 3, 3, 1), act_func='lrelu', modulate=False, g_arch_len=18 * 4):
        super().__init__()
        self.out_channel = out_channel
        self.conv1 = ConvLayer(in_channel, in_channel, 3, activate=act_func, modulate=modulate, g_arch_len=g_arch_len)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, blur_kernel=blur_kernel, activate=act_func, modulate=modulate, g_arch_len=g_arch_len)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate='none', bias=False, modulate=modulate, g_arch_len=g_arch_len)

    def forward(self, x, g_arch=None):
        out = self.conv1(x, g_arch)
        out = self.conv2(out, g_arch)
        skip = self.skip(x, g_arch)
        out = (out + skip) / math.sqrt(2)
        return out


class Discriminator(nn.Module):

    def __init__(self, resolution, channel_multiplier=2, channel_max=512, blur_kernel=(1, 3, 3, 1), act_func='lrelu'):
        super().__init__()
        channels = {(4): 4096, (8): 2048, (16): 1024, (32): 512, (64): 256, (128): 128, (256): 64, (512): 32, (1024): 16}
        channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in channels.items()}
        convs = [ConvLayer(3, channels[resolution], 1, activate=act_func)]
        log_res = int(math.log(resolution, 2))
        in_channel = channels[resolution]
        for i in range(log_res, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, act_func=act_func))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, activate=act_func)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation=act_func), EqualLinear(channels[4], 1))

    def forward(self, x):
        out = self.convs(x)
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


D_CHANNEL_CONFIG = G_CHANNEL_CONFIG


class DiscriminatorMultiRes(nn.Module):

    def __init__(self, resolution, channel_multiplier=2, channel_max=512, blur_kernel=(1, 3, 3, 1), act_func='lrelu', n_res=1, modulate=False):
        super().__init__()
        channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in D_CHANNEL_CONFIG.items()}
        self.convs = nn.ModuleList()
        self.res2idx = {}
        for i_res in range(n_res):
            cur_res = resolution // 2 ** i_res
            self.res2idx[cur_res] = i_res
            self.convs.append(ConvLayer(3, channels[cur_res], 1, activate=act_func))
        log_res = int(math.log(resolution, 2))
        in_channel = channels[resolution]
        self.blocks = nn.ModuleList()
        for i in range(log_res, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.blocks.append(ResBlock(in_channel, out_channel, blur_kernel, act_func=act_func, modulate=modulate and i in list(range(log_res, 2, -1))[-2:], g_arch_len=4 * (log_res * 2 - 2)))
            in_channel = out_channel
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, activate=act_func)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation=act_func), EqualLinear(channels[4], 1))

    def forward(self, x, g_arch=None):
        res = x.shape[-1]
        idx = self.res2idx[res]
        out = self.convs[idx](x)
        for i in range(idx, len(self.blocks)):
            out = self.blocks[i](out, g_arch)
        out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out).view(out.shape[0], -1)
        out = self.final_linear(out)
        return out

    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out


class ResNet50Encoder(nn.Module):

    def __init__(self, n_style, style_dim=512, mean_latent=None, pretrained=False):
        super().__init__()
        self.n_style = n_style
        self.style_dim = style_dim
        model_tmp = resnet50(pretrained=pretrained)
        self.conv1 = model_tmp.conv1
        self.bn1 = model_tmp.bn1
        self.relu = model_tmp.relu
        self.maxpool = model_tmp.maxpool
        self.layer1 = model_tmp.layer1
        self.layer2 = model_tmp.layer2
        self.layer3 = model_tmp.layer3
        self.layer4 = model_tmp.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(model_tmp.fc.in_features, n_style * style_dim)
        self.register_buffer('mean_latent', torch.rand(style_dim))
        if mean_latent is not None:
            self.mean_latent.data.copy_(mean_latent)

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.n_style, self.style_dim) + self.mean_latent.view(1, 1, -1).detach()
        return x


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


def safe_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    try:
        world_size = hvd.size()
    except:
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    if world_size == 1:
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    else:
        if hvd.rank() == 0:
            _ = torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
        hvd.broadcast(torch.tensor(0), root_rank=0, name='dummy')
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)


def fid_inception_v3():
    """Build pretrained Inception models for FID computation

    The Inception models for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception models.
    """
    import torchvision
    inception = torchvision.models.Inception3(num_classes=1008, aux_logits=False, init_weights=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = safe_load_state_dict_from_url(FID_WEIGHTS_URL, progress=True, map_location='cpu')
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=False, normalize_input=True, requires_grad=False, use_fid_inception=True):
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
            feeding input to models. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the models require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception models used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception models
            available in torchvision. The FID Inception models has different
            weights and a slightly different structure from torchvision's
            Inception models. If you want to compute FID scores, you are
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvLayer,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EqualConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
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
     True),
    (InceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (ModulatedConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyledConv,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ToRGB,
     lambda: ([], {'in_channel': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_mit_han_lab_anycost_gan(_paritybench_base):
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

