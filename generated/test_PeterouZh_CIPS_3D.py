import sys
_module = sys.modules[__name__]
del sys
start_modelarts_v2 = _module
diffaug = _module
discriminator = _module
generator = _module
generator_v1 = _module
multi_head_mapping = _module
st_web = _module
eval_fid = _module
gen_images = _module
sample_images = _module
setup_evaluation = _module
train = _module
discriminator = _module
generator = _module
generator_v1 = _module
generator_v2 = _module
generator_v3 = _module
generator_v4 = _module
generator_v5 = _module
st_web = _module
gen_images = _module
train = _module
comm_model_utils = _module
comm_utils = _module
diff_aug = _module
models = _module
cond_layer_norm = _module
fc_net = _module
film_layer = _module
inr_network = _module
mod_conv_fc = _module
multi_head_mapping = _module
nerf_network = _module
op = _module
fused_act = _module
upfirdn2d = _module
generator_conv_nerf = _module
generators = _module
generators_3D2D = _module
dev = _module
nerf_inr = _module
curriculums = _module
fid_evaluation = _module
cond_layer_norm = _module
discriminator_v10 = _module
discriminator_v11 = _module
discriminator_v15 = _module
discriminator_v9 = _module
generator_nerf_inr = _module
generator_nerf_inr_v1 = _module
generator_nerf_inr_v10 = _module
generator_nerf_inr_v11 = _module
generator_nerf_inr_v12 = _module
generator_nerf_inr_v13 = _module
generator_nerf_inr_v14 = _module
generator_nerf_inr_v15 = _module
generator_nerf_inr_v16 = _module
generator_nerf_inr_v16_ablation = _module
generator_nerf_inr_v2 = _module
generator_nerf_inr_v3 = _module
generator_nerf_inr_v4 = _module
generator_nerf_inr_v5 = _module
generator_nerf_inr_v6 = _module
generator_nerf_inr_v8 = _module
generator_nerf_inr_v9 = _module
generator_pigan_v16 = _module
stylegan_disc_v16 = _module
eval_v16 = _module
save_v16 = _module
train = _module
train_pigan_v16 = _module
train_v10 = _module
train_v12 = _module
train_v13 = _module
train_v14 = _module
train_v15 = _module
train_v16 = _module
train_v7 = _module
train_v9 = _module
datasets = _module
generators = _module
math_utils_torch = _module
siren = _module
volumetric_rendering = _module
pigan_model_utils = _module
pigan_utils = _module
extract_shapes = _module
inverse_render = _module
render_multiview_images = _module
render_video = _module
render_video_interpolation = _module
train = _module
test_cips3d = _module
test_cips3d_inversion = _module
datasets = _module
discriminators = _module
sgdiscriminators = _module
eval_metrics = _module
extract_shapes = _module
fid_evaluation = _module
generators = _module
math_utils_torch = _module
volumetric_rendering = _module
inverse_render = _module
render_multiview_images = _module
render_video = _module
render_video_interpolation = _module
siren = _module
train = _module
dataset_tool = _module
plot_fid = _module
web_demo = _module

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


import torch.nn.functional as F


import collections


import logging


import math


from torch import nn


from itertools import chain


from collections import OrderedDict


import random


import time


import numpy as np


import torch.nn as nn


from torch.cuda.amp import autocast


import copy


from torchvision import transforms


import torchvision.transforms.functional as trans_f


from torchvision.utils import save_image


from torchvision.utils import make_grid


import torch.distributed as dist


import torch.utils.data as data_utils


import torchvision.transforms as tv_trans


from functools import partial


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


import torchvision.transforms.functional as tv_f


import itertools


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.cpp_extension import load


from collections import deque


import torchvision.transforms as tv_transforms


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import torchvision


from torchvision import datasets


import torchvision.transforms as transforms


import matplotlib.pyplot as plt


import scipy


from torch.nn.utils import spectral_norm


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
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


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

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConvTranspose2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, upsample=False, padding='zero'):
        layers = collections.OrderedDict()
        self.padding = 0
        stride = 1
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers['down_blur'] = Blur(blur_kernel, pad=(pad0, pad1))
            stride = 2
        if upsample:
            up_conv = EqualConvTranspose2d(in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate)
            layers['up_conv'] = up_conv
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            layers['up_blur'] = Blur(blur_kernel, pad=(pad0, pad1))
        else:
            if not downsample:
                if padding == 'zero':
                    self.padding = (kernel_size - 1) // 2
                elif padding == 'reflect':
                    padding = (kernel_size - 1) // 2
                    if padding > 0:
                        layers['pad'] = nn.ReflectionPad2d(padding)
                    self.padding = 0
                elif padding != 'valid':
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')
            equal_conv = EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate)
            layers['equal_conv'] = equal_conv
        if activate:
            if bias:
                layers['flrelu'] = FusedLeakyReLU(out_channel)
            else:
                layers['slrelu'] = ScaledLeakyReLU(0.2)
        super().__init__(layers)
        pass


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], kernel_size=3, downsample=True, first_downsample=False):
        super().__init__()
        if first_downsample:
            self.conv1 = ConvLayer(in_channel, in_channel, kernel_size, downsample=downsample)
            self.conv2 = ConvLayer(in_channel, out_channel, kernel_size)
        else:
            self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
            self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        """

    :param in_dim:
    :param out_dim:
    :param bias:
    :param bias_init:
    :param lr_mul: 0.01
    :param activation: None: Linear; fused_leaky_relu
    """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        if self.activation is not None:
            self.act_layer = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul
        pass

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        if self.activation:
            out = self.act_layer(out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}), activation={self.activation}'


class Discriminator(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, **kwargs):
        super().__init__()
        self.repr = f'size={size}, channel_multiplier={channel_multiplier}, n_first_layers={n_first_layers}'
        self.size = size
        self.input_size = input_size
        self.epoch = 0
        self.step = 0
        channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        convs = []
        _conv_layer = ConvLayer(input_size, channels[size], 1)
        convs.append(_conv_layer)
        _first_layers = [ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)]
        convs.extend(_first_layers)
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
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))
        torch_utils.print_number_params(models_dict={'convs': self.convs, 'final_conv': self.final_conv, 'final_linear': self.final_linear, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, *args, **kwargs):
        assert input.shape[-1] == self.size
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.convs, inputs_args=(input,), submodels=['0', '1'], name_prefix='convs.')
        out = self.convs(input)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_conv, inputs_args=(out,), name_prefix='final_conv.')
        out = self.final_conv(out)
        out = out.view(batch, -1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_linear, inputs_args=(out,), name_prefix='final_linear.')
        out = self.final_linear(out)
        latent, position = None, None
        return out, latent, position


class Discriminator_MultiScale(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, max_size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, first_downsample=False, channels=None, **kwargs):
        super().__init__()
        self.repr = f'max_size={max_size}, channel_multiplier={channel_multiplier}, n_first_layers={n_first_layers},first_downsample={first_downsample}'
        self.epoch = 0
        self.step = 0
        self.max_size = max_size
        self.input_size = input_size
        if channels is None:
            channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.conv_in = nn.ModuleDict()
        for name, channel_ in channels.items():
            self.conv_in[f'{name}'] = ConvLayer(input_size, channel_, 1)
        self.convs = nn.ModuleDict()
        log_size = int(math.log(max_size, 2))
        in_channel = channels[max_size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs[f'{2 ** i}'] = ResBlock(in_channel, out_channel, blur_kernel, first_downsample=first_downsample)
            in_channel = out_channel
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))
        torch_utils.print_number_params(models_dict={'conv_in': self.conv_in, 'convs': self.convs, 'final_conv': self.final_conv, 'final_linear': self.final_linear, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, **kwargs):
        size = input.shape[-1]
        log_size = int(math.log(size, 2))
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.conv_in[f'{2 ** log_size}'], inputs_args=(input,), name_prefix=f'conv_in[{2 ** log_size}].')
        cur_size_out = self.conv_in[f'{2 ** log_size}'](input)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.convs[f'{2 ** log_size}'], inputs_args=(cur_size_out,), name_prefix=f'convs[{2 ** log_size}].')
        cur_size_out = self.convs[f'{2 ** log_size}'](cur_size_out)
        if alpha < 1:
            down_input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.conv_in[f'{2 ** (log_size - 1)}'], inputs_args=(down_input,), name_prefix=f'conv_in[{2 ** (log_size - 1)}].')
            down_size_out = self.conv_in[f'{2 ** (log_size - 1)}'](down_input)
            out = alpha * cur_size_out + (1 - alpha) * down_size_out
        else:
            out = cur_size_out
        for i in range(log_size - 1, 2, -1):
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.convs[f'{2 ** i}'], inputs_args=(out,), name_prefix=f'convs[{2 ** i}].')
            out = self.convs[f'{2 ** i}'](out)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_conv, inputs_args=(out,), name_prefix='final_conv.')
        out = self.final_conv(out)
        out = out.view(batch, -1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_linear, inputs_args=(out,), name_prefix='final_linear.')
        out = self.final_linear(out)
        latent, position = None, None
        return out, latent, position


class Discriminator_MultiScale_Aux(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, max_size, channel_multiplier=2, first_downsample=False, **kwargs):
        super().__init__()
        self.repr = f'max_size={max_size}, channel_multiplier={channel_multiplier}, first_downsample={first_downsample}'
        self.epoch = 0
        self.step = 0
        self.main_disc = Discriminator_MultiScale(max_size=max_size, channel_multiplier=channel_multiplier, first_downsample=first_downsample)
        channel_multiplier = 2
        channels = {(4): 128 * channel_multiplier, (8): 128 * channel_multiplier, (16): 128 * channel_multiplier, (32): 128 * channel_multiplier, (64): 128 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.aux_disc = Discriminator_MultiScale(max_size=max_size, channel_multiplier=channel_multiplier, first_downsample=True, channels=channels)
        torch_utils.print_number_params(models_dict={'main_disc': self.main_disc, 'aux_disc': self.aux_disc, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, use_aux_disc=False, **kwargs):
        if use_aux_disc:
            b = input.shape[0] // 2
            main_input = input[:b]
            aux_input = input[b:]
            main_out, latent, position = self.main_disc(main_input, alpha, **kwargs)
            aux_out, _, _ = self.aux_disc(aux_input, alpha, **kwargs)
            out = torch.cat([main_out, aux_out], dim=0)
        else:
            out, latent, position = self.main_disc(input, alpha, **kwargs)
        return out, latent, position


class SkipLayer(nn.Module):

    def __init__(self):
        super(SkipLayer, self).__init__()

    def forward(self, x0, x1):
        out = x0 + x1
        return out


class SinAct(nn.Module):

    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LinearSinAct(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearSinAct, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sin = SinAct()
        pass

    def forward(self, x, *args, **kwargs):
        x = self.linear(x)
        x = self.sin(x)
        return x


class FiLMLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class NeRFNetwork(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim=3, hidden_dim=256, hidden_layers=2, style_dim=512, rgb_dim=3, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, style_dim={style_dim}, rgb_dim={rgb_dim}, '
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.final_layer.apply(film_layer.frequency_init(25))
        _in_dim = hidden_dim
        _out_dim = hidden_dim
        self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim), nn.Tanh())
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = nerf_network.UniformBoxWarp(0.24)
        torch_utils.print_number_params({'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
            x = layer(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        style = style_dict[f'{self.name_prefix}_rgb']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_sine, inputs_args=(x, style), name_prefix=f'color_layer_sine.')
        x = self.color_layer_sine(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        out = self.forward_with_frequencies_phase_shifts(input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs)
        return out

    def print_number_params(self):
        None
        pass

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(self, transformed_points, transformed_ray_directions_expanded, style_dict, max_points, num_steps):
        batch_size, num_points, _ = transformed_points.shape
        rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1), device=self.device)
        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b:b + 1, head:tail] = self(input=transformed_points[b:b + 1, head:tail], style_dict={name: style[b:b + 1] for name, style in style_dict.items()}, ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                head += max_points
        rgb_sigma_output = rearrange(rgb_sigma_output, 'b (hw s) rgb_sigma -> b hw s rgb_sigma', s=num_steps)
        return rgb_sigma_output


class NeRFNetwork_sigma(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim=3, hidden_dim=256, hidden_layers=2, style_dim=512, rgb_dim=3, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, style_dim={style_dim}, rgb_dim={rgb_dim}, '
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            if True:
                _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            else:
                _layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = 3
        _out_dim = hidden_dim // 2
        self.color_layer_sine = LinearSinAct(in_features=_in_dim, out_features=_out_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = nerf_network.UniformBoxWarp(0.24)
        torch_utils.print_number_params({'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
            x = layer(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_sine, inputs_args=(input,), name_prefix=f'color_layer_sine.')
        x = self.color_layer_sine(input)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        out = self.forward_with_frequencies_phase_shifts(input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs)
        return out

    def print_number_params(self):
        None
        pass

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(self, transformed_points, transformed_ray_directions_expanded, style_dict, max_points, num_steps):
        batch_size, num_points, _ = transformed_points.shape
        rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1), device=self.device)
        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b:b + 1, head:tail] = self(input=transformed_points[b:b + 1, head:tail], style_dict={name: style[b:b + 1] for name, style in style_dict.items()}, ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                head += max_points
        rgb_sigma_output = rearrange(rgb_sigma_output, 'b (hw s) rgb_sigma -> b hw s rgb_sigma', s=num_steps)
        return rgb_sigma_output


class INRNetwork_Skip(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, input_dim, style_dim, hidden_layers, dim_scale=1, rgb_dim=3, device=None, name_prefix='inr', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'input_dim={input_dim}, style_dim={style_dim}, hidden_layers={hidden_layers}, dim_scale={dim_scale}, '
        self.device = device
        self.rgb_dim = rgb_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.channels = {(0): int(512 * dim_scale), (1): int(512 * dim_scale), (2): int(512 * dim_scale), (3): int(512 * dim_scale), (4): int(512 * dim_scale), (5): int(128 * dim_scale), (6): int(64 * dim_scale), (7): int(32 * dim_scale), (8): int(16 * dim_scale)}
        self.style_dim_dict = {}
        _out_dim = input_dim
        self.network = nn.ModuleList()
        self.to_rbgs = nn.ModuleList()
        for i in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = self.channels[i]
            _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{i}_0'] = _layer.style_dim
            _layer = film_layer.FiLMLayer(in_dim=_out_dim, out_dim=_out_dim, style_dim=style_dim)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{i}_1'] = _layer.style_dim
            to_rgb = inr_network.ToRGB(in_dim=_out_dim, dim_rgb=3)
            self.to_rbgs.append(to_rgb)
        self.tanh = nn.Sequential(nn.Tanh())
        torch_utils.print_number_params({'network': self.network, 'to_rbgs': self.to_rbgs, 'inr_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, style_dict, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        rgb = 0
        for index in range(self.hidden_layers):
            _layer = self.network[index * 2]
            style = style_dict[f'{self.name_prefix}_w{index}_0']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(_layer, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.network.{index}.0.')
            x = _layer(x, style)
            _layer = self.network[index * 2 + 1]
            style = style_dict[f'{self.name_prefix}_w{index}_1']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(_layer, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.network.{index}.1.')
            x = _layer(x, style)
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.to_rbgs[index], inputs_args=(x, rgb), name_prefix=f'to_rgb.{index}')
            rgb = self.to_rbgs[index](x, skip=rgb)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class ModSinLayer(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim, use_style_fc=False, style_dim=None, which_linear=nn.Linear, spectral_norm=False, eps=1e-05, freq=1, phase=0, **kwargs):
        super(ModSinLayer, self).__init__()
        self.repr = f'in_dim={in_dim}, use_style_fc={use_style_fc}, style_dim={style_dim}, freq={freq}, phase={phase}'
        self.in_dim = in_dim
        self.use_style_fc = use_style_fc
        self.style_dim = style_dim
        self.freq = freq
        self.phase = phase
        self.spectral_norm = spectral_norm
        if use_style_fc:
            self.gain_fc = which_linear(style_dim, in_dim)
            self.bias_fc = which_linear(style_dim, in_dim)
            if spectral_norm:
                self.gain_fc = nn.utils.spectral_norm(self.gain_fc)
                self.bias_fc = nn.utils.spectral_norm(self.bias_fc)
        else:
            self.style_dim = in_dim * 2
        self.eps = eps
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        pass

    def forward(self, x, style):
        """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        assert style.shape[-1] == self.style_dim
        if self.use_style_fc:
            gain = self.gain_fc(style) + 1.0
            bias = self.bias_fc(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
            gain = gain + 1.0
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-08)
        x = x * gain + bias
        out = self.lrelu(x)
        return out


class ModSinLayer_NoBias(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim, use_style_fc=False, style_dim=None, which_linear=nn.Linear, spectral_norm=False, eps=1e-05, freq=1, phase=0, **kwargs):
        super(ModSinLayer_NoBias, self).__init__()
        self.repr = f'in_dim={in_dim}, use_style_fc={use_style_fc}, style_dim={style_dim}, freq={freq}, phase={phase}'
        self.in_dim = in_dim
        self.use_style_fc = use_style_fc
        self.style_dim = style_dim
        self.freq = freq
        self.phase = phase
        self.spectral_norm = spectral_norm
        if use_style_fc:
            self.gain_fc = which_linear(style_dim, in_dim)
            if spectral_norm:
                self.gain_fc = nn.utils.spectral_norm(self.gain_fc)
        else:
            self.style_dim = in_dim * 2
        self.eps = eps
        pass

    def forward(self, x, style):
        """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        assert style.shape[-1] == self.style_dim
        if self.use_style_fc:
            gain = self.gain_fc(style) + 1.0
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
            gain = gain + 1.0
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        x = torch.sin(self.freq * x + self.phase)
        out = x * gain
        return out


class SinBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.mod1 = mod_conv_fc.SinStyleMod(in_channel=in_dim, out_channel=out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_0'] = self.mod1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.mod2 = mod_conv_fc.SinStyleMod(in_channel=out_dim, out_channel=out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_1'] = self.mod2.style_dim
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.skip = SkipLayer()
        pass

    def forward(self, x, style_dict, skip=False):
        x_orig = x
        style = style_dict[f'{self.name_prefix}_0']
        x = self.mod1(x, style)
        x = self.act1(x)
        style = style_dict[f'{self.name_prefix}_1']
        x = self.mod2(x, style)
        out = self.act2(x)
        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = self.skip(out, x_orig)
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim})'
        return repr


class ToRGB(nn.Module):

    def __init__(self, in_dim, dim_rgb=3, use_equal_fc=False):
        super().__init__()
        self.in_dim = in_dim
        self.dim_rgb = dim_rgb
        if use_equal_fc:
            self.linear = mod_conv_fc.EqualLinear(in_dim, dim_rgb, scale=1.0)
        else:
            self.linear = nn.Linear(in_dim, dim_rgb)
        pass

    def forward(self, input, skip=None):
        out = self.linear(input)
        if skip is not None:
            out = out + skip
        return out


class CIPSNet(nn.Module):

    def __repr__(self):
        return tl2_utils.get_class_repr(self)

    def __init__(self, input_dim, style_dim, hidden_dim=256, pre_rgb_dim=32, device=None, name_prefix='inr', **kwargs):
        """

    :param input_dim:
    :param style_dim:
    :param hidden_dim:
    :param pre_rgb_dim:
    :param device:
    :param name_prefix:
    :param kwargs:
    """
        super().__init__()
        self.repr_str = tl2_utils.dict2string(dict_obj={'input_dim': input_dim, 'style_dim': style_dim, 'hidden_dim': hidden_dim, 'pre_rgb_dim': pre_rgb_dim})
        self.device = device
        self.pre_rgb_dim = pre_rgb_dim
        self.name_prefix = name_prefix
        self.channels = {'4': hidden_dim, '8': hidden_dim, '16': hidden_dim, '32': hidden_dim, '64': hidden_dim, '128': hidden_dim, '256': hidden_dim, '512': hidden_dim, '1024': hidden_dim}
        self.module_name_list = []
        self.style_dim_dict = {}
        _out_dim = input_dim
        network = OrderedDict()
        to_rbgs = OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel
            if name.startswith(('none',)):
                _linear_block = inr_network.LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = SinBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block
            _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=pre_rgb_dim, use_equal_fc=False)
            to_rbgs[name] = _to_rgb
        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(inr_network.frequency_init(100))
        self.module_name_list.append('network')
        self.module_name_list.append('to_rgbs')
        out_layers = []
        if pre_rgb_dim > 3:
            out_layers.append(nn.Linear(pre_rgb_dim, 3))
        out_layers.append(nn.Tanh())
        self.tanh = nn.Sequential(*out_layers)
        self.tanh.apply(inr_network.frequency_init(100))
        self.module_name_list.append('tanh')
        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['cips'] = self
        logger = logging.getLogger('tl')
        torch_utils.print_number_params(models_dict=models_dict, logger=logger)
        logger.info(self)
        pass

    def forward(self, input, style_dict, img_size=1024, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        img_size = str(2 ** int(np.log2(img_size)))
        rgb = 0
        for idx, (name, block) in enumerate(self.network.items()):
            if idx >= 4:
                skip = True
            else:
                skip = False
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(block, inputs_args=(x, style_dict, skip), submodels=['mod1', 'mod2'], name_prefix=f'block.{name}.')
            x = block(x, style_dict, skip=skip)
            if idx >= 3:
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(self.to_rgbs[name], inputs_args=(x, rgb), name_prefix=f'to_rgb.{name}.')
                rgb = self.to_rgbs[name](x, skip=rgb)
            if name == img_size:
                break
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


def _kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MultiHeadMappingNetwork(nn.Module):

    def __init__(self, z_dim, hidden_dim, base_layers, head_layers, head_dim_dict):
        super().__init__()
        self.head_dim_dict = head_dim_dict
        base_net = []
        for i in range(base_layers):
            if i == 0:
                in_dim = z_dim
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim
            base_net.append(nn.Linear(in_dim, out_dim))
            if i != base_layers - 1:
                base_net.append(nn.LeakyReLU(0.2, inplace=True))
        self.base_net = nn.Sequential(*base_net)
        self.base_net.apply(_kaiming_leaky_init)
        for name, head_dim in head_dim_dict.items():
            head_net = []
            in_dim = hidden_dim
            for i in range(head_layers):
                if i == head_layers - 1:
                    out_dim = head_dim
                else:
                    out_dim = hidden_dim
                head_net.append(nn.Linear(in_dim, out_dim))
                if i != head_layers - 1:
                    head_net.append(nn.LeakyReLU(0.2, inplace=True))
            self.add_module(name, nn.Sequential(*head_net))
            with torch.no_grad():
                getattr(self, name)[-1].weight *= 0.25
        self.print_number_params()
        pass

    def forward(self, z):
        base_fea = self.base_net(z)
        out_dict = {}
        for name, head_dim in self.head_dim_dict.items():
            head_net = getattr(self, name)
            out = head_net(base_fea)
            out_dict[name] = out
        return out_dict

    def print_number_params(self):
        models_dict = {'base_net': self.base_net}
        for name, _ in self.head_dim_dict.items():
            models_dict[name] = getattr(self, name)
        models_dict['mapping_network'] = self
        None
        torch_utils.print_number_params(models_dict)
        pass


class MultiHeadMappingNetwork_EqualLR(nn.Module):

    def __init__(self, z_dim, hidden_dim, base_layers, head_layers, head_dim_dict, use_equal_fc=True, lr_mlp=1.0, scale=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.head_dim_dict = head_dim_dict
        out_dim = z_dim
        self.repr = f'z_dim={z_dim}, hidden_dim={hidden_dim}, base_layers={base_layers}, head_layers={head_layers}, use_equal_fc={use_equal_fc}, scale={scale}'
        self.norm = PixelNorm()
        base_net = []
        for i in range(base_layers):
            in_dim = out_dim
            out_dim = hidden_dim
            if use_equal_fc:
                base_layer_ = EqualLinear(in_dim=in_dim, out_dim=out_dim, lr_mul=lr_mlp, scale=scale)
            else:
                base_layer_ = nn.Linear(in_features=in_dim, out_features=out_dim)
                base_layer_.apply(_kaiming_leaky_init)
            base_net.append(base_layer_)
            if head_layers > 0 or i != base_layers - 1:
                act_layer_ = nn.LeakyReLU(0.2, inplace=True)
                base_net.append(act_layer_)
        if len(base_net) > 0:
            self.base_net_elr = nn.Sequential(*base_net)
            self.num_z = 1
        else:
            self.base_net_elr = None
            self.num_z = len(head_dim_dict)
        head_in_dim = out_dim
        for name, head_dim in head_dim_dict.items():
            if head_layers > 0:
                head_net = []
                out_dim = head_in_dim
                for i in range(head_layers):
                    in_dim = out_dim
                    if i == head_layers - 1:
                        out_dim = head_dim
                    else:
                        out_dim = hidden_dim
                    if use_equal_fc:
                        head_layer_ = EqualLinear(in_dim=in_dim, out_dim=out_dim, lr_mul=lr_mlp)
                    else:
                        head_layer_ = nn.Linear(in_features=in_dim, out_features=out_dim)
                        head_layer_.apply(_kaiming_leaky_init)
                    head_net.append(head_layer_)
                    if i != head_layers - 1:
                        act_layer_ = nn.LeakyReLU(0.2, inplace=True)
                        head_net.append(act_layer_)
                head_net = nn.Sequential(*head_net)
            else:
                head_net = nn.Identity()
            self.add_module(name, head_net)
        models_dict = {'base_net': self.base_net_elr}
        for name, _ in self.head_dim_dict.items():
            models_dict[name] = getattr(self, name)
        models_dict['mapping_network'] = self
        torch_utils.print_number_params(models_dict)
        logging.getLogger('tl').info(self)
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def forward(self, z):
        """

    :param z:
    :return:
    """
        if self.base_net_elr is not None:
            z = self.norm(z)
            base_fea = self.base_net_elr(z)
            head_inputs = {name: base_fea for name in self.head_dim_dict.keys()}
        else:
            head_inputs = {}
            for idx, name in enumerate(self.head_dim_dict.keys()):
                head_inputs[name] = self.norm(z[idx])
        out_dict = {}
        for name, head_dim in self.head_dim_dict.items():
            head_net = getattr(self, name)
            head_input_ = head_inputs[name]
            out = head_net(head_input_)
            out_dict[name] = out
        return out_dict


class Generator_Diffcam(nn.Module):

    def __repr__(self):
        return tl2_utils.get_class_repr(self)

    def __init__(self, nerf_cfg, mapping_shape_cfg, mapping_app_cfg, inr_cfg, mapping_inr_cfg, shape_block_end_index=None, app_block_end_index=None, inr_block_end_index=None, device='cuda', inr_detach=False, **kwargs):
        super(Generator_Diffcam, self).__init__()
        self.repr_str = tl2_utils.dict2string(dict_obj={'nerf_cfg': nerf_cfg, 'mapping_shape_cfg': mapping_shape_cfg, 'mapping_app_cfg': mapping_app_cfg, 'inr_cfg': inr_cfg, 'mapping_inr_cfg': mapping_inr_cfg, 'shape_block_end_index': shape_block_end_index, 'app_block_end_index': app_block_end_index, 'inr_block_end_index': inr_block_end_index, 'inr_detach': inr_detach})
        self.device = device
        self.inr_block_end_index = inr_block_end_index
        self.inr_detach = inr_detach
        self.module_name_list = []
        self.nerf_net = pigan_net.piGAN_NeRF_Net(shape_block_end_index=shape_block_end_index, app_block_end_index=app_block_end_index, **nerf_cfg)
        self.module_name_list.append('nerf_net')
        self.mapping_shape = multi_head_mapping.MultiHeadMappingNetwork(**{**mapping_shape_cfg, 'head_dim_dict': self.nerf_net.style_dim_dict_shape})
        self.module_name_list.append('mapping_shape')
        self.mapping_app = multi_head_mapping.MultiHeadMappingNetwork(**{**mapping_app_cfg, 'head_dim_dict': self.nerf_net.style_dim_dict_app})
        self.module_name_list.append('mapping_app')
        _in_dim = self.nerf_net.out_dim
        self.inr_net = cips_net.CIPSNet(**{**inr_cfg, 'input_dim': _in_dim, 'add_out_layer': True})
        self.module_name_list.append('inr_net')
        self.mapping_inr = multi_head_mapping.MultiHeadMappingNetwork(**{**mapping_inr_cfg, 'head_dim_dict': self.inr_net.style_dim_dict})
        self.module_name_list.append('mapping_inr')
        self.aux_to_rbg = nn.Sequential(nn.Linear(_in_dim, 3), nn.Tanh())
        self.aux_to_rbg.apply(nerf_network.frequency_init(25))
        self.module_name_list.append('aux_to_rbg')
        logger = logging.getLogger('tl')
        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['G'] = self
        torch_utils.print_number_params(models_dict=models_dict, logger=logger)
        logger.info(self)
        pass

    def get_subnet_grad_norm(self):
        ret_dict = {}
        for name in self.module_name_list:
            subnet = getattr(self, name)
            grad_norm = torch_utils.get_grad_norm_total(params=subnet.parameters())
            ret_dict[f'G_GN.{name}'] = grad_norm
        return ret_dict

    def forward(self, zs, rays_o, rays_d, nerf_kwargs={}, psi=1, return_aux_img=False, grad_points=None, forward_points=None, **kwargs):
        """
    Generates images from a noise vector, rendering parameters, and camera distribution.
    Uses the hierarchical sampling scheme described in NeRF.

    :param zs: {k: (b, z_dim), ...}
    :param rays_o: (b, h, w, 3) in world space
    :param rays_d: (b, h, w, 3) in world space

    :return:
    - pixels: (b, 3, h, w)
    - pitch_yaw: (b, 2)
    """
        style_dict = self.mapping_network(**zs)
        if psi < 1:
            avg_styles = self.generate_avg_frequencies(device=self.device)
            style_dict = self.get_truncated_freq_phase(raw_style_dict=style_dict, avg_style_dict=avg_styles, raw_lambda=psi)
        b, h, w, c = rays_o.shape
        rays_o = rearrange(rays_o, 'b h w c -> b (h w) c')
        rays_d = rearrange(rays_d, 'b h w c -> b (h w) c')
        if grad_points is not None and grad_points < h * w:
            imgs, ret_maps = self.part_grad_forward(rays_o=rays_o, rays_d=rays_d, style_dict=style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img, grad_points=grad_points)
        else:
            imgs, ret_maps = self.whole_grad_forward(rays_o=rays_o, rays_d=rays_d, style_dict=style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img, forward_points=forward_points)
        imgs = rearrange(imgs, 'b (h w) c -> b c h w', h=h, w=w)
        ret_imgs = {}
        for name, v_map in ret_maps.items():
            if v_map.dim() == 3:
                v_map = rearrange(v_map, 'b (h w) c -> b c h w', h=h, w=w)
            elif v_map.dim() == 2:
                v_map = rearrange(v_map, 'b (h w) -> b h w', h=h, w=w)
            ret_imgs[name] = v_map
        return imgs, ret_imgs

    def get_rays_axis_angle(self, R, t, fx, fy, H: int, W: int, N_rays: int=-1):
        """

    :param R: (b, 3)
    :param t: (b, 3)
    :param fx:
    :param fy:
    :param H:
    :param W:
    :param N_rays:
    :return

    - rays_o: (b, H, W, 3)
    - rays_d: (b, H, W, 3)
    - select_inds: (b, H, W)
    """
        rays_o, rays_d, select_inds = cam_params.get_rays(rot=R, trans=t, focal_x=fx, focal_y=fy, H=H, W=W, N_rays=N_rays, flatten=False)
        return rays_o, rays_d, select_inds

    def get_batch_style_dict(self, b, style_dict):
        ret_style_dict = {}
        for name, style in style_dict.items():
            ret_style_dict[name] = style[[b]]
        return ret_style_dict

    def whole_grad_forward(self, rays_o, rays_d, style_dict, nerf_kwargs, return_aux_img=True, forward_points=None, **kwargs):
        if forward_points is not None and forward_points < rays_o.shape[1]:
            with torch.no_grad():
                batch_size = rays_o.shape[0]
                num_points = rays_o.shape[1]
                near = nerf_kwargs['near']
                far = nerf_kwargs['far']
                N_samples = nerf_kwargs['N_samples']
                perturb = self.training
                z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=N_samples, perturb=perturb)
                batch_image_ddict = collections.defaultdict(list)
                for b in range(batch_size):
                    image_ddict = collections.defaultdict(list)
                    head = 0
                    while head < num_points:
                        tail = head + forward_points
                        cur_style_dict = self.get_batch_style_dict(b=b, style_dict=style_dict)
                        cur_inr_img, cur_ret_maps = self.points_forward(rays_o=rays_o[[b], head:tail], rays_d=rays_d[[b], head:tail], points=points[[b], head:tail], z_vals=z_vals[[b], head:tail], style_dict=cur_style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img)
                        image_ddict['inr_img'].append(cur_inr_img)
                        for k, v in cur_ret_maps.items():
                            image_ddict[k].append(v)
                        head += forward_points
                    for k, v in image_ddict.items():
                        one_image = torch.cat(v, dim=1)
                        batch_image_ddict[k].append(one_image)
                ret_maps = {}
                for k, v in batch_image_ddict.items():
                    ret_maps[k] = torch.cat(v, dim=0)
                imgs = ret_maps.pop('inr_img')
        else:
            near = nerf_kwargs['near']
            far = nerf_kwargs['far']
            N_samples = nerf_kwargs['N_samples']
            perturb = self.training
            z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=N_samples, perturb=perturb)
            imgs, ret_maps = self.points_forward(rays_o=rays_o, rays_d=rays_d, points=points, z_vals=z_vals, style_dict=style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img)
        return imgs, ret_maps

    def part_grad_forward(self, rays_o, rays_d, style_dict, nerf_kwargs, return_aux_img, grad_points):
        near = nerf_kwargs['near']
        far = nerf_kwargs['far']
        N_samples = nerf_kwargs['N_samples']
        perturb = self.training
        z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o, rays_d=rays_d, near=near, far=far, N_samples=N_samples, perturb=perturb)
        batch_size = rays_o.shape[0]
        num_points = rays_o.shape[1]
        device = self.device
        assert num_points > grad_points
        idx_grad, idx_no_grad = torch_utils.batch_random_split_indices(bs=batch_size, num_points=num_points, grad_points=grad_points, device=device)
        inr_img_grad, ret_maps_grad = self.points_forward(rays_o=rays_o, rays_d=rays_d, points=points, z_vals=z_vals, style_dict=style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img, idx_grad=idx_grad)
        with torch.no_grad():
            inr_img_no_grad, ret_maps_no_grad = self.points_forward(rays_o=rays_o, rays_d=rays_d, points=points, z_vals=z_vals, style_dict=style_dict, nerf_kwargs=nerf_kwargs, return_aux_img=return_aux_img, idx_grad=idx_no_grad)
        imgs = comm_utils.batch_scatter_points(idx_grad=idx_grad, points_grad=inr_img_grad, idx_no_grad=idx_no_grad, points_no_grad=inr_img_no_grad, num_points=num_points)
        ret_maps = {}
        for k in ret_maps_grad.keys():
            comp_map = comm_utils.batch_scatter_points(idx_grad=idx_grad, points_grad=ret_maps_grad[k], idx_no_grad=idx_no_grad, points_no_grad=ret_maps_no_grad[k], num_points=num_points)
            ret_maps[k] = comp_map
        return imgs, ret_maps

    def points_forward(self, rays_o, rays_d, points, z_vals, style_dict, nerf_kwargs, return_aux_img, idx_grad=None, **kwargs):
        """

    :param rays_o: (b, hxw, 3)
    :param rays_d: (b, hxw, 3)
    :param points: (b, hxw, Nsamples, 3)
    :param z_vals: (b, hxw, Nsamples)
    :param style_dict:
    :param nerf_kwargs:
    :param return_aux_img:
    :param idx_grad: (b, N_grad, )
    :param kwargs:
    :return:
    """
        device = points.device
        viewdirs = volume_rendering.get_viewdirs(rays_d=rays_d)
        N_samples = nerf_kwargs['N_samples']
        if idx_grad is not None:
            rays_o = comm_utils.batch_gather_points(points=rays_o, idx_grad=idx_grad)
            rays_d = comm_utils.batch_gather_points(points=rays_d, idx_grad=idx_grad)
            points = comm_utils.batch_gather_points(points=points, idx_grad=idx_grad)
            z_vals = comm_utils.batch_gather_points(points=z_vals, idx_grad=idx_grad)
        points = rearrange(points, 'b Nrays Nsamples c -> b (Nrays Nsamples) c')
        coarse_viewdirs = repeat(viewdirs, 'b Nrays c -> b (Nrays Nsamples) c', Nsamples=N_samples)
        coarse_output = self.nerf_net(x=points, ray_directions=coarse_viewdirs, style_dict=style_dict)
        padd = 0.0
        mask_box = torch.all(points <= 1.0 + padd, dim=-1) & torch.all(points >= -1.0 - padd, dim=-1)
        coarse_output[mask_box == 0] = 0.0
        coarse_output = rearrange(coarse_output, 'b (Nrays Nsamples) rgb_sigma -> b Nrays Nsamples rgb_sigma', Nsamples=N_samples)
        if nerf_kwargs['N_importance'] > 0:
            with torch.no_grad():
                raw_sigma = coarse_output[..., -1]
                perturb = self.training
                fine_z_vals, fine_points = volume_rendering.get_fine_points(z_vals=z_vals, rays_o=rays_o, rays_d=rays_d, raw_sigma=raw_sigma, N_importance=nerf_kwargs['N_importance'], perturb=perturb, raw_noise_std=nerf_kwargs['raw_noise_std'], eps=nerf_kwargs['eps'])
            fine_points = rearrange(fine_points, 'b Nrays Nsamples c -> b (Nrays Nsamples) c')
            fine_viewdirs = repeat(viewdirs, 'b Nrays c -> b (Nrays Nsamples) c', Nsamples=nerf_kwargs['N_importance'])
            fine_output = self.nerf_net(x=fine_points, ray_directions=fine_viewdirs, style_dict=style_dict)
            fine_output = rearrange(fine_output, 'b (Nrays Nsamples) rgb_sigma -> b Nrays Nsamples rgb_sigma', Nsamples=nerf_kwargs['N_importance'])
            DIM_SAMPLES = 2
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=DIM_SAMPLES)
            _, indices = torch.sort(all_z_vals, dim=DIM_SAMPLES)
            all_z_vals = torch.gather(all_z_vals, DIM_SAMPLES, indices)
            all_outputs = torch.cat([fine_output, coarse_output], dim=DIM_SAMPLES)
            view_shape = [*indices.shape, *((len(all_outputs.shape) - len(indices.shape)) * [1])]
            all_outputs = torch.gather(all_outputs, DIM_SAMPLES, indices.view(view_shape).expand_as(all_outputs))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        all_raw_rgb = all_outputs[..., :-1]
        all_raw_sigma = all_outputs[..., -1]
        pixels_fea, ret_maps = volume_rendering.ray_integration(raw_rgb=all_raw_rgb, raw_sigma=all_raw_sigma, z_vals=all_z_vals, rays_d=rays_d, raw_noise_std=nerf_kwargs['raw_noise_std'], eps=nerf_kwargs['eps'])
        if self.inr_detach:
            inr_img = self.inr_net(pixels_fea.detach(), style_dict, block_end_index=self.inr_block_end_index)
        else:
            inr_img = self.inr_net(pixels_fea, style_dict, block_end_index=self.inr_block_end_index)
        if return_aux_img:
            aux_img = self.aux_to_rbg(pixels_fea)
            ret_maps['aux_img'] = aux_img
        return inr_img, ret_maps

    def z_sampler(self, shape, device, dist='gaussian'):
        if dist == 'gaussian':
            z = torch.randn(shape, device=device)
        elif dist == 'uniform':
            z = torch.rand(shape, device=device) * 2 - 1
        return z

    def get_zs(self, b, batch_split=1):
        z_shape = self.z_sampler(shape=(b, self.mapping_shape.z_dim), device=self.device)
        z_app = self.z_sampler(shape=(b, self.mapping_app.z_dim), device=self.device)
        z_inr = self.z_sampler(shape=(b, self.mapping_inr.z_dim), device=self.device)
        if batch_split > 1:
            zs_list = []
            z_shape_list = z_shape.split(b // batch_split)
            z_app_list = z_app.split(b // batch_split)
            z_inr_list = z_inr.split(b // batch_split)
            for z_shape_, z_app_, z_inr_ in zip(z_shape_list, z_app_list, z_inr_list):
                zs_ = {'z_shape': z_shape_, 'z_app': z_app_, 'z_inr': z_inr_}
                zs_list.append(zs_)
            return zs_list
        else:
            zs = {'z_shape': z_shape, 'z_app': z_app, 'z_inr': z_inr}
            return zs

    def mapping_network(self, z_shape, z_app, z_inr):
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.mapping_shape, inputs_args=(z_shape,), submodels=['base_net'], name_prefix='mapping_shape.')
            VerboseModel.forward_verbose(self.mapping_app, inputs_args=(z_app,), submodels=['base_net'], name_prefix='mapping_app.')
            VerboseModel.forward_verbose(self.mapping_inr, inputs_args=(z_inr,), submodels=['base_net'], input_padding=50, name_prefix='mapping_inr.')
        style_dict = {}
        style_dict.update(self.mapping_shape(z_shape))
        style_dict.update(self.mapping_app(z_app))
        style_dict.update(self.mapping_inr(z_inr))
        return style_dict

    def get_truncated_freq_phase(self, raw_style_dict, avg_style_dict, raw_lambda):
        truncated_style_dict = {}
        for name, avg_style in avg_style_dict.items():
            raw_style = raw_style_dict[name]
            truncated_style = avg_style + raw_lambda * (raw_style - avg_style)
            truncated_style_dict[name] = truncated_style
        return truncated_style_dict

    def generate_avg_frequencies(self, num_samples=10000, device='cuda'):
        """Calculates average frequencies and phase shifts"""
        zs = self.get_zs(num_samples)
        with torch.no_grad():
            style_dict = self.mapping_network(**zs)
        avg_styles = {}
        for name, style in style_dict.items():
            avg_styles[name] = style.mean(0, keepdim=True)
        return avg_styles

    def staged_forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_device(self, device):
        pass

    def forward_camera_pos_and_lookup(self, zs, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, camera_pos, camera_lookup, psi=1, sample_dist=None, lock_view_dependence=False, clamp_mode='relu', nerf_noise=0.0, white_back=False, last_back=False, return_aux_img=False, grad_points=None, forward_points=None, **kwargs):
        """
    Generates images from a noise vector, rendering parameters, and camera distribution.
    Uses the hierarchical sampling scheme described in NeRF.

    :param z: (b, z_dim)
    :param img_size:
    :param fov: face: 12
    :param ray_start: face: 0.88
    :param ray_end: face: 1.12
    :param num_steps: face: 12
    :param h_stddev: face: 0.3
    :param v_stddev: face: 0.155
    :param h_mean: face: pi/2
    :param v_mean: face: pi/2
    :param hierarchical_sample: face: true
    :param camera_pos: (b, 3)
    :param camera_lookup: (b, 3)
    :param psi: [0, 1]
    :param sample_dist: mode for sample_camera_positions, face: 'gaussian'
    :param lock_view_dependence: face: false
    :param clamp_mode: face: 'relu'
    :param nerf_noise:
    :param last_back: face: false
    :param white_back: face: false
    :param kwargs:
    :return:
    - pixels: (b, 3, h, w)
    - pitch_yaw: (b, 2)
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.mapping_network_nerf, inputs_args=(zs['z_nerf'],), submodels=['base_net'], name_prefix='mapping_nerf.')
            VerboseModel.forward_verbose(self.mapping_network_inr, inputs_args=(zs['z_inr'],), submodels=['base_net'], input_padding=50, name_prefix='mapping_inr.')
        style_dict = self.mapping_network(**zs)
        if psi < 1:
            avg_styles = self.generate_avg_frequencies(device=self.device)
            style_dict = self.get_truncated_freq_phase(raw_style_dict=style_dict, avg_style_dict=avg_styles, raw_lambda=psi)
        if grad_points is not None and grad_points < img_size ** 2:
            imgs, pitch_yaw = self.part_grad_forward(style_dict=style_dict, img_size=img_size, fov=fov, ray_start=ray_start, ray_end=ray_end, num_steps=num_steps, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, hierarchical_sample=hierarchical_sample, sample_dist=sample_dist, lock_view_dependence=lock_view_dependence, clamp_mode=clamp_mode, nerf_noise=nerf_noise, white_back=white_back, last_back=last_back, return_aux_img=return_aux_img, grad_points=grad_points, camera_pos=camera_pos, camera_lookup=camera_lookup)
            return imgs, pitch_yaw
        else:
            imgs, pitch_yaw = self.whole_grad_forward(style_dict=style_dict, img_size=img_size, fov=fov, ray_start=ray_start, ray_end=ray_end, num_steps=num_steps, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, hierarchical_sample=hierarchical_sample, sample_dist=sample_dist, lock_view_dependence=lock_view_dependence, clamp_mode=clamp_mode, nerf_noise=nerf_noise, white_back=white_back, last_back=last_back, return_aux_img=return_aux_img, forward_points=forward_points, camera_pos=camera_pos, camera_lookup=camera_lookup)
            return imgs, pitch_yaw


class Generator_Diffcam_FreezeNeRF(Generator_Diffcam):

    def load_nerf_ema(self, G_ema):
        ret = self.nerf_net.load_state_dict(G_ema.nerf_net.state_dict())
        ret = self.mapping_shape.load_state_dict(G_ema.mapping_shape.state_dict())
        ret = self.mapping_app.load_state_dict(G_ema.mapping_app.state_dict())
        ret = self.aux_to_rbg.load_state_dict(G_ema.aux_to_rbg.state_dict())
        pass

    def forward(self, **kwargs):
        self.nerf_net.requires_grad_(False)
        self.mapping_shape.requires_grad_(False)
        self.mapping_app.requires_grad_(False)
        self.aux_to_rbg.requires_grad_(False)
        return super(Generator_Diffcam_FreezeNeRF, self).forward(**kwargs)

    def mapping_network(self, z_shape, z_app, z_inr):
        style_dict = {}
        with torch.no_grad():
            style_dict.update(self.mapping_shape(z_shape))
            style_dict.update(self.mapping_app(z_app))
        style_dict.update(self.mapping_inr(z_inr))
        return style_dict


class PosEmbedding(nn.Module):

    def __init__(self, max_logscale, N_freqs, logscale=True, multi_pi=False):
        """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
    """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        if logscale:
            self.freqs = 2 ** torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2 ** max_logscale, N_freqs)
        if multi_pi:
            self.freqs = self.freqs * math.pi
        pass

    def get_out_dim(self):
        outdim = 3 + 3 * 2 * self.N_freqs
        return outdim

    def forward(self, x):
        """
    Inputs:
        x: (B, 3)

    Outputs:
        out: (B, 6*N_freqs+3)
    """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


class CLN(nn.Module):

    def __init__(self, in_dim, use_style_fc=False, style_dim=None, which_linear=nn.Linear, spectral_norm=False, eps=1e-05, **kwargs):
        super(CLN, self).__init__()
        self.in_dim = in_dim
        self.use_style_fc = use_style_fc
        self.style_dim = style_dim
        self.spectral_norm = spectral_norm
        if use_style_fc:
            self.gain = which_linear(style_dim, in_dim)
            self.bias = which_linear(style_dim, in_dim)
            if spectral_norm:
                self.gain = nn.utils.spectral_norm(self.gain)
                self.bias = nn.utils.spectral_norm(self.bias)
        else:
            self.style_dim = in_dim * 2
        self.eps = eps
        pass

    def forward(self, x, style):
        """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        if self.use_style_fc:
            gain = self.gain(style) + 1.0
            bias = self.bias(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
            gain = gain + 1.0
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        out = F.layer_norm(x, normalized_shape=(self.in_dim,), weight=None, bias=None, eps=self.eps)
        out = out * gain + bias
        return out

    def __repr__(self):
        s = f'{self.__class__.__name__}(in_dim={self.in_dim}, style_dim={self.style_dim})'
        return s


class FCNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_layers, rgb_dim=3, device=None, name_prefix='fc', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        _out_dim = input_dim
        network = []
        for i in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            network.append(nn.Linear(in_features=_in_dim, out_features=_out_dim))
            network.append(nn.LeakyReLU(0.2, inplace=True))
        if len(network) > 0:
            self.network = nn.Sequential(*network)
            self.network.apply(init_func.kaiming_leaky_init)
        else:
            self.network = nn.Identity()
        self.to_rbg = nn.Sequential(nn.Linear(_out_dim, rgb_dim), nn.Tanh())
        torch_utils.print_number_params({'network': self.network, 'to_rbg': self.to_rbg, 'fc_net': self})
        pass

    def forward(self, input, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param kwargs:
    :return:

    """
        x = input
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.network, inputs_args=(x,), name_prefix=f'{self.name_prefix}.network.')
        x = self.network(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.to_rbg, inputs_args=(x,), name_prefix='to_rgb.')
        out = self.to_rbg(x)
        return out


class LinearScale(nn.Module):

    def __init__(self, scale, bias):
        super(LinearScale, self).__init__()
        self.scale_v = scale
        self.bias_v = bias
        pass

    def forward(self, x):
        out = x * self.scale_v + self.bias_v
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(scale_v={self.scale_v},bias_v={self.bias_v})'
        return repr


class FiLMLayer_PreSin(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, use_style_fc=True, which_linear=nn.Linear, **kwargs):
        super(FiLMLayer_PreSin, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.linear = which_linear(in_dim, out_dim)
        nn.init.uniform_(self.linear.weight, -np.sqrt(9 / in_dim), np.sqrt(9 / in_dim))
        if use_style_fc:
            self.gain_fc = which_linear(style_dim, out_dim)
            self.bias_fc = which_linear(style_dim, out_dim)
            self.gain_fc.weight.data.mul_(0.25)
            self.gain_fc.bias.data.fill_(1)
            self.bias_fc.weight.data.mul_(0.25)
        else:
            self.style_dim = out_dim * 2
        pass

    def forward(self, x, style):
        """

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        if self.use_style_fc:
            gain = self.gain_fc(style)
            bias = self.bias_fc(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        x = self.linear(x)
        x = torch.sin(x)
        out = gain * x + bias
        return out

    def __repr__(self):
        s = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim}, use_style_fc={self.use_style_fc}, )'
        return s


class INRNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_layers, fc_layers, rgb_dim=3, device=None, name_prefix='inr', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        for i in range(hidden_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim
            mod_fc = pigan_model_utils.Modulated_FC_Conv(in_channel=in_dim, out_channel=out_dim, activation='FusedLeakyReLU')
            self.network.append(mod_fc)
        if len(self.network) > 0:
            self.style_dim_dict[f'{name_prefix}_network'] = len(self.network) * mod_fc.style_dim
        else:
            out_dim = input_dim
        self.fc_net = nn.ModuleList()
        for i in range(fc_layers):
            if i == 0:
                in_dim = out_dim
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim
            fc_layer = pigan_model_utils.EqualLinear(in_dim=in_dim, out_dim=out_dim, activation='fused_leaky_relu')
            self.fc_net.append(fc_layer)
        self.to_rbg = nn.Sequential(pigan_model_utils.EqualLinear(in_dim=out_dim, out_dim=rgb_dim, activation=None), nn.Tanh())
        self.dim_styles = sum(self.style_dim_dict.values())
        self.print_number_params()
        pass

    def forward(self, input, style_dict, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        if len(self.network) > 0:
            style = style_dict[f'{self.name_prefix}_network']
            for index, layer in enumerate(self.network):
                start = index * layer.style_dim
                end = (index + 1) * layer.style_dim
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(layer, inputs_args=(x, style[..., start:end]), name_prefix=f'{self.name_prefix}.network.{index}.')
                x = layer(x, style[..., start:end])
        if len(self.fc_net) > 0:
            for index, layer in enumerate(self.fc_net):
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(layer, inputs_args=(x,), name_prefix=f'{self.name_prefix}.fc_net.{index}.')
                x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.to_rbg, inputs_args=(x,), name_prefix='to_rgb.')
        out = self.to_rbg(x)
        return out

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def print_number_params(self):
        None
        torch_utils.print_number_params({'network': self.network, 'fc_net': self.fc_net, 'to_rbg': self.to_rbg, 'inr_net': self})
        pass


class FiLMBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.film1 = FiLMLayer(in_dim=in_dim, out_dim=out_dim, style_dim=style_dim)
        self.style_dim_dict[f'{name_prefix}_0'] = self.film1.style_dim
        self.film2 = FiLMLayer(in_dim=out_dim, out_dim=out_dim, style_dim=style_dim)
        self.style_dim_dict[f'{name_prefix}_1'] = self.film2.style_dim
        pass

    def forward(self, x, style_dict, skip=False):
        x_orig = x
        style = style_dict[f'{self.name_prefix}_0']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.film1, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.film1.')
        x = self.film1(x, style)
        style = style_dict[f'{self.name_prefix}_1']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.film2, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.film2.')
        out = self.film2(x, style)
        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = out + x_orig
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim})'
        return repr


class LinearBlock(nn.Module):

    def __init__(self, in_dim, out_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name_prefix = name_prefix
        self.net = nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(in_features=out_dim, out_features=out_dim), nn.LeakyReLU(0.2, inplace=True))
        self.net.apply(init_func.kaiming_leaky_init)
        pass

    def forward(self, x, *args, **kwargs):
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.net, inputs_args=(x,), name_prefix=f'{self.name_prefix}.net.')
        out = self.net(x)
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim})'
        return repr


def frequency_init(freq):

    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class INRNetwork_Skip_Prog(nn.Module):

    def __init__(self, input_dim, style_dim, dim_scale=1, rgb_dim=3, device=None, name_prefix='inr', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.device = device
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        self.channels = {'16': int(256 * dim_scale), '32': int(256 * dim_scale), '64': int(256 * dim_scale), '128': int(256 * dim_scale), '256': int(256 * dim_scale), '512': int(256 * dim_scale), '1024': int(256 * dim_scale)}
        self.style_dim_dict = {}
        _out_dim = input_dim
        network = collections.OrderedDict()
        to_rbgs = collections.OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel
            if name.startswith(('none',)):
                _linear_block = LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = FiLMBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block
            _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=3)
            to_rbgs[name] = _to_rgb
        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(frequency_init(100))
        self.tanh = nn.Sequential(nn.Tanh())
        self.dim_styles = sum(self.style_dim_dict.values())
        torch_utils.print_number_params({'network': self.network, 'to_rbgs': self.to_rgbs, 'inr_net': self})
        pass

    def forward(self, input, style_dict, img_size, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        img_size = str(2 ** int(np.log2(img_size)))
        rgb = 0
        for name, block in self.network.items():
            x = block(x, style_dict, skip=True)
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.to_rgbs[name], inputs_args=(x, rgb), name_prefix=f'to_rgb.{name}.')
            rgb = self.to_rgbs[name](x, skip=rgb)
            if name == img_size:
                break
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class CLNBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
        self.style_dim_dict[f'{name_prefix}_0'] = self.cln1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.cln2 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
        self.style_dim_dict[f'{name_prefix}_1'] = self.cln2.style_dim
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        pass

    def forward(self, x, style_dict, skip=False):
        x_orig = x
        x = self.linear1(x)
        style = style_dict[f'{self.name_prefix}_0']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.cln1, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.cln1.')
        x = self.cln1(x, style)
        x = self.act1(x)
        x = self.linear2(x)
        style = style_dict[f'{self.name_prefix}_1']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.cln2, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.cln2.')
        x = self.cln2(x, style)
        out = self.act2(x)
        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = out + x_orig
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim})'
        return repr


class AddLayer(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x0, x1):
        return x0 + x1


class ModLinearBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.linear1 = mod_conv_fc.Modulated_FC_Conv(in_channel=in_dim, out_channel=out_dim, style_dim=style_dim, use_style_fc=True, scale=None, eps=0.0001)
        self.style_dim_dict[f'{name_prefix}_0'] = self.linear1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = mod_conv_fc.Modulated_FC_Conv(in_channel=out_dim, out_channel=out_dim, style_dim=style_dim, use_style_fc=True, scale=None, eps=0.0001)
        self.style_dim_dict[f'{name_prefix}_1'] = self.linear2.style_dim
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.add_layer = AddLayer()
        pass

    def forward(self, x, style_dict, skip=False):
        x_orig = x
        style = style_dict[f'{self.name_prefix}_0']
        x = self.linear1(x, style)
        x = self.act1(x)
        style = style_dict[f'{self.name_prefix}_1']
        x = self.linear2(x, style)
        out = self.act2(x)
        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = self.add_layer(out, x_orig)
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim})'
        return repr


class ModToRGB(nn.Module):

    def __init__(self, in_dim, dim_rgb=3):
        super().__init__()
        self.in_dim = in_dim
        self.dim_rgb = dim_rgb
        self.linear = mod_conv_fc.EqualLinear(in_dim=in_dim, out_dim=dim_rgb)
        pass

    def forward(self, input, skip=None):
        out = self.linear(input)
        if skip is not None:
            out = out + skip
        return out

    def __repr__(self):
        repr = f'{self.__class__.__name__}(in_dim={self.in_dim}, dim_rgb={self.dim_rgb})'
        return repr


class INRNetwork_Skip_CLN(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, input_dim, style_dim, hidden_dim=256, rgb_dim=3, device=None, name_prefix='inr', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'input_dim={input_dim}, style_dim={style_dim}, hidden_dim={hidden_dim}'
        self.device = device
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        self.channels = {'4': hidden_dim, '8': hidden_dim, '16': hidden_dim, '32': hidden_dim, '64': hidden_dim, '128': hidden_dim, '256': hidden_dim, '512': hidden_dim, '1024': hidden_dim}
        self.style_dim_dict = {}
        _out_dim = input_dim
        network = OrderedDict()
        to_rbgs = OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel
            if name.startswith(('none',)):
                _linear_block = inr_network.LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = ModLinearBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block
            _to_rgb = ModToRGB(in_dim=_out_dim, dim_rgb=3)
            to_rbgs[name] = _to_rgb
        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(inr_network.frequency_init(100))
        self.tanh = nn.Sequential(nn.Tanh())
        self.dim_styles = sum(self.style_dim_dict.values())
        torch_utils.print_number_params({'network': self.network, 'to_rbgs': self.to_rgbs, 'inr_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, style_dict, img_size, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        img_size = str(2 ** int(np.log2(img_size)))
        rgb = 0
        for name, block in self.network.items():
            skip = int(name) >= 32
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(block, inputs_args=(x, style_dict, skip), name_prefix=f'block.{name}.')
            x = block(x, style_dict, skip=skip)
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.to_rgbs[name], inputs_args=(x, rgb), name_prefix=f'to_rgb.{name}.')
            rgb = self.to_rgbs[name](x, skip=rgb)
            if name == img_size:
                break
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class CLNLayer(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim, out_dim, style_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.repr = f'in_dim={in_dim}, out_dim={out_dim}, style_dim={style_dim}'
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
        self.style_dim = self.cln1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        pass

    def forward(self, x, style):
        x = self.linear1(x)
        x = self.cln1(x, style)
        x = self.act1(x)
        return x


class Linear_Skip_Prog(nn.Module):

    def __init__(self, input_dim, hidden_dim, rgb_dim=3, device=None, name_prefix='linear', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        self.channels = {'32': int(hidden_dim), '64': int(hidden_dim), '128': int(hidden_dim), '256': int(hidden_dim), '512': int(hidden_dim), '1024': int(hidden_dim)}
        _in_dim = input_dim
        _out_dim = hidden_dim
        self.cln_layer = CLNLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=256, name_prefix=f'{name_prefix}_w')
        self.style_dim_dict = self.cln_layer.style_dim_dict
        network = collections.OrderedDict()
        to_rbgs = collections.OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel
            _linear_block = nn.Sequential(nn.Linear(in_features=_in_dim, out_features=_out_dim), nn.LeakyReLU(0.2, inplace=True))
            network[name] = _linear_block
            _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=self.rgb_dim)
            to_rbgs[name] = _to_rgb
        self.network = nn.ModuleDict(network)
        self.network.apply(frequency_init(25))
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.tanh = nn.Sequential(nn.Tanh())
        torch_utils.print_number_params({'network': self.network, 'to_rbgs': self.to_rgbs, 'linear_net': self})
        pass

    def forward(self, input, img_size, style_dict, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        img_size = str(2 ** int(np.log2(img_size)))
        x = self.cln_layer(x, style_dict)
        rgb = 0
        for name, block in self.network.items():
            x = block(x)
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(self.to_rgbs[name], inputs_args=(x, rgb), name_prefix=f'to_rgb.{name}.')
            rgb = self.to_rgbs[name](x, skip=rgb)
            if name == img_size:
                break
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class Modulated_FC_Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=1, style_dim=None, use_style_fc=False, demodulate=True, activation=None):
        """

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param style_dim: =in_channel
    :param use_style_fc:
    :param demodulate:
    :param activation: FusedLeakyReLU
    """
        super().__init__()
        self.eps = 1e-08
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.demodulate = demodulate
        self.activation = activation
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        if use_style_fc:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=0)
        else:
            self.style_dim = in_channel
        if activation is not None:
            self.act_layer = nn.LeakyReLU(0.2)
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, use_style_fc={self.use_style_fc}, activation={self.activation})'

    def forward(self, x, style):
        """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
        if x.dim() == 2:
            input = rearrange(x, 'b c -> b c 1 1')
        elif x.dim() == 3:
            input = rearrange(x, 'b n c -> b c n 1')
        elif x.dim() == 4:
            input = x
        else:
            assert 0
        batch, in_channel, height, width = input.shape
        if self.use_style_fc:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            style = style + 1.0
        else:
            style = rearrange(style, 'b c -> b 1 c 1 1')
            style = style + 1.0
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        out = out + self.bias
        if self.activation is not None:
            out = self.act_layer(out)
        if x.dim() == 2:
            out = rearrange(out, 'b c 1 1 -> b c')
        elif x.dim() == 3:
            out = rearrange(out, 'b c n 1 -> b n c')
        return out


class SinStyleMod(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_channel, out_channel, kernel_size=1, style_dim=None, use_style_fc=False, demodulate=True, use_group_conv=False, eps=1e-08, **kwargs):
        """

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param style_dim: =in_channel
    :param use_style_fc:
    :param demodulate:
    """
        super().__init__()
        self.eps = eps
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.demodulate = demodulate
        self.use_group_conv = use_group_conv
        self.padding = kernel_size // 2
        if use_group_conv:
            self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        else:
            assert kernel_size == 1
            self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if use_style_fc:
            self.modulation = nn.Linear(style_dim, in_channel)
            self.modulation.apply(init_func.kaiming_leaky_init)
        else:
            self.style_dim = in_channel
        self.sin = SinAct()
        self.norm = nn.LayerNorm(in_channel)
        self.repr = f'in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, style_dim={style_dim}, use_style_fc={use_style_fc}, demodulate={demodulate}, use_group_conv={use_group_conv}'
        pass

    def forward_bmm(self, x, style, weight):
        """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, 'b c -> b 1 c')
        elif x.dim() == 3:
            input = x
        else:
            assert 0
        batch, N, in_channel = input.shape
        if self.use_style_fc:
            style = self.modulation(style)
            style = style.view(-1, in_channel, 1)
        else:
            style = rearrange(style, 'b c -> b c 1')
        weight = weight * (style + 1)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([1]) + self.eps)
            weight = weight * demod.view(batch, 1, self.out_channel)
        out = torch.bmm(input, weight)
        if x.dim() == 2:
            out = rearrange(out, 'b 1 c -> b c')
        elif x.dim() == 3:
            pass
        return out

    def forward_group_conv(self, x, style):
        """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, 'b c -> b c 1 1')
        elif x.dim() == 3:
            input = rearrange(x, 'b n c -> b c n 1')
        elif x.dim() == 4:
            input = x
        else:
            assert 0
        batch, in_channel, height, width = input.shape
        if self.use_style_fc:
            style = self.modulation(style).view(-1, 1, in_channel, 1, 1)
            style = style + 1.0
        else:
            style = rearrange(style, 'b c -> b 1 c 1 1')
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        if x.dim() == 2:
            out = rearrange(out, 'b c 1 1 -> b c')
        elif x.dim() == 3:
            out = rearrange(out, 'b c n 1 -> b n c')
        return out

    def forward(self, x, style, force_bmm=False):
        """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
        if self.use_group_conv:
            if force_bmm:
                weight = rearrange(self.weight, '1 out in 1 1 -> 1 in out')
                out = self.forward_bmm(x=x, style=style, weight=weight)
            else:
                out = self.forward_group_conv(x=x, style=style)
        else:
            out = self.forward_bmm(x=x, style=style, weight=self.weight)
        return out


class UniformBoxWarp(nn.Module):

    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class NeRFNetworkL(NeRFNetwork):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, in_dim=3, hidden_dim=256, rgb_dim=3, style_dim=512, hidden_layers=2, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super(NeRFNetwork, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            _film_layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_film_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim
        _out_dim = hidden_dim
        self.color_layer_sine = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = UniformBoxWarp(0.24)
        self.print_number_params()
        pass


class NeRFNetwork_CLN(NeRFNetwork):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, in_dim=3, hidden_dim=256, rgb_dim=3, style_dim=512, hidden_layers=2, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super(NeRFNetwork, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            if idx == 0:
                _film_layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
                self.network.append(_film_layer)
                self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
            else:
                _cln_layer = CLNLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_w{idx}')
                self.network.append(_cln_layer)
                self.style_dim_dict.update(_cln_layer.style_dim_dict)
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim
        _out_dim = hidden_dim
        self.color_layer_sine = CLNLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_rgb')
        self.style_dim_dict.update(self.color_layer_sine.style_dim_dict)
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = UniformBoxWarp(0.24)
        self.print_number_params()
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            if index == 0:
                style = style_dict[f'{self.name_prefix}_w{index}']
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
                x = layer(x, style)
            else:
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(layer, inputs_args=(x, style_dict), name_prefix=f'network.{index}.')
                x = layer(x, style_dict)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_sine, inputs_args=(x, style_dict), name_prefix=f'color_layer_sine.')
        x = self.color_layer_sine(x, style_dict)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out


class NeRFNetwork_Small(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim=3, hidden_dim=256, rgb_dim=3, style_dim=512, hidden_layers=2, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super(NeRFNetwork_Small, self).__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, rgb_dim={rgb_dim}, style_dim={style_dim}, hidden_layers={hidden_layers}'
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.network = nn.ModuleList()
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim
        _out_dim = hidden_dim
        self.color_layer_sine = nn.Identity()
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.gridwarper = UniformBoxWarp(0.24)
        torch_utils.print_number_params({'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
            x = layer(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        out = self.forward_with_frequencies_phase_shifts(input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs)
        return out


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SynthesisNetwork(nn.Module):

    def __init__(self, hidden_dim, num_conv_synthesis, kernel_size=3):
        super(SynthesisNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        layers = []
        in_conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=(kernel_size, kernel_size), padding=padding, padding_mode='reflect')
        layers.append(('in_conv', in_conv))
        act_layer = FiLMLayer()
        layers.append((f'act_in', act_layer))
        for idx in range(num_conv_synthesis - 1):
            conv_layer = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(kernel_size, kernel_size), padding=padding, padding_mode='reflect')
            layers.append((f'conv_{idx}', conv_layer))
            act_layer = FiLMLayer()
            layers.append((f'act_{idx}', act_layer))
        self.network = nn.ModuleDict(layers)
        self.network.apply(frequency_init(25))
        in_conv.apply(first_layer_film_sine_init)
        pass

    def forward(self, x, frequencies, phase_shifts, img_size):
        """
    x: (b, hxw, step, 3)

    out: (b, h x w x step, c)
    """
        num_steps = x.shape[-2]
        x = rearrange(x, 'b (h w) n c -> (b n) c h w', h=img_size)
        index = 0
        for name, layer in self.network.items():
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            if isinstance(layer, FiLMLayer):
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(model=layer, inputs_args=(x, frequencies[..., start:end], phase_shifts[..., start:end]), name_prefix=name, register_children=False, register_itself=True, input_padding=35)
                x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
                index += 1
            else:
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(model=layer, inputs_args=(x,), name_prefix=name, register_children=False, register_itself=True, input_padding=35)
                x = layer(x)
        x = rearrange(x, '(b n) c h w -> b (h w n) c', n=num_steps)
        if global_cfg.tl_debug:
            torch_utils.print_number_params(models_dict={'synthesis network': self})
        return x


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):

    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_output_dim))
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]
        return frequencies, phase_shifts


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(3, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([rbg, sigma], dim=-1)


def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""
    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 10000000000.0 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)
    noise = torch.randn(sigmas.shape, device=device) * noise_std
    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * F.softplus(sigmas + noise))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * F.relu(sigmas + noise))
    else:
        raise 'Need to choose clamp mode'
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)
    if last_back:
        weights[:, :, -1] += 1 - weights_sum
    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    if white_back:
        rgb_final = rgb_final + 1 - weights_sum
    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1.0, 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)
    return rgb_final, depth_final, weights


def normalize_vecs(vectors: torch.Tensor) ->torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""
    W, H = resolution
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device), torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan(2 * math.pi * fov / 360 / 2)
    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W * H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals
    points = torch.stack(n * [points])
    z_vals = torch.stack(n * [z_vals])
    rays_d_cam = torch.stack(n * [rays_d_cam])
    return points, z_vals, rays_d_cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-05):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))
    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:, :, 1:2, :] - z_vals[:, :, 0:1, :]
    offset = (torch.rand(z_vals.shape, device=device) - 0.5) * distance_between_points
    z_vals = z_vals + offset
    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi * 0.5, vertical_mean=math.pi * 0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean
    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean
    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = (torch.rand((n, 1), device=device) - 0.5) * 2 * v_stddev + v_mean
        v = torch.clamp(v, 1e-05, 1 - 1e-05)
        phi = torch.arccos(1 - 2 * v)
    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean
    phi = torch.clamp(phi, 1e-05, math.pi - 1e-05)
    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r * torch.cos(phi)
    return output_points, phi, theta


def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal'):
    """Samples a camera position and maps points in camera space to world space."""
    n, num_rays, num_steps, channels = points.shape
    points, z_vals = perturb_points(points, z_vals, ray_directions, device)
    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)
    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)
    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, num_rays, 3)
    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


class ImplicitGenerator3d(nn.Module):

    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device
        self.generate_avg_frequencies()

    def forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        batch_size = z.shape[0]
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
        coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
            fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        return pixels, torch.cat([pitch, yaw], -1)

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""
        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts

    def staged_forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """
        batch_size = z.shape[0]
        self.generate_avg_frequencies()
        with torch.no_grad():
            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)
            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                        head += max_batch_size
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode=kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
        return pixels, depth_map

    def staged_forward_with_frequencies(self, truncated_frequencies, truncated_phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        batch_size = truncated_frequencies.shape[0]
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                        head += max_batch_size
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode=kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
        return pixels, depth_map

    def forward_with_frequencies(self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        batch_size = frequencies.shape[0]
        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        return pixels, torch.cat([pitch, yaw], -1)


class Embedding3D(nn.Module):

    def __init__(self, channel=128, size=64):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(1, channel, size, size, size))
        pass

    def forward(self, points, x_scale, y_scale, z_scale, **kwargs):
        """
        points: (b, hxwxd, 3) xyz
        point_shape: [b, h, w, d, 3]
        """
        b = points.shape[0]
        emb = self.emb.expand(b, -1, -1, -1, -1)
        xyz_scale = torch.tensor([x_scale, y_scale, z_scale], device=points.device)
        points = points * xyz_scale
        points = rearrange(points, 'b n xyz -> b n 1 1 xyz')
        out = F.grid_sample(emb, points, padding_mode='border', mode='bilinear', align_corners=False)
        out = rearrange(out, 'b c n 1 1 -> b n c')
        return out


class SinActivation(nn.Module):

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CoordFC(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        nn.init.uniform_(self.layer.weight, -np.sqrt(9 / input_dim), np.sqrt(9 / input_dim))
        self.act = SinActivation()
        pass

    def forward(self, x):
        x = self.layer(x)
        out = self.act(x)
        return out


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ResidualCCBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p), nn.LeakyReLU(0.2, inplace=True), CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class AdapterBlock(nn.Module):

    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, output_channels, 1, padding=0), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.model(input)


class CCSEncoderDiscriminator(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCCBlock(32, 64), ResidualCCBlock(64, 128), ResidualCCBlock(128, 256), ResidualCCBlock(256, 400), ResidualCCBlock(400, 400), ResidualCCBlock(400, 400), ResidualCCBlock(400, 400)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:257]
        position = x[..., 257:259]
        return prediction, latent, position


class ResidualCCBlockFirstDown(nn.Module):

    def __init__(self, inplanes, planes, stride=2, kernel_size=3, spectral_norm=True, skip=True):
        super().__init__()
        self.skip = skip
        p = kernel_size // 2
        conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=p, stride=stride)
        conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=p)
        if spectral_norm:
            conv2 = nn.utils.spectral_norm(conv2)
        self.network = nn.Sequential(conv1, nn.LeakyReLU(0.2, inplace=True), conv2, nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        if skip:
            proj = nn.Conv2d(inplanes, planes, 1, stride=stride)
            if spectral_norm:
                proj = nn.utils.spectral_norm(proj)
            self.proj = proj
        else:
            self.proj = nn.Identity()
        pass

    def forward(self, input):
        y = self.network(input)
        if self.skip:
            identity = self.proj(input)
        else:
            identity = 0
        y = y + identity
        return y


class DiscriminatorMultiScale(nn.Module):

    def __init__(self, dim_z=256, spectral_norm=False, **kwargs):
        """
    # from 4 * 2^0 to 4 * 2^7 4 -> 512

    :param kwargs:
    """
        super().__init__()
        self.dim_z = dim_z
        self.spectral_norm = spectral_norm
        logger = logging.getLogger('tl')
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCCBlockFirstDown(3, 16, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(16, 32, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(32, 64, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(64, 128, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(128, 256, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(256, 512, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm), ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm)])
        self.layers.apply(init_func.kaiming_leaky_init)
        final_layer = nn.Linear(512, 1 + dim_z + 2)
        self.final_layer = final_layer
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'final_layer': self.final_layer, 'D': self}, logger=logger)
        logger.info(self)
        pass

    def forward(self, x, alpha, **kwargs):
        img_size = x.shape[-1]
        if img_size < 128:
            x = F.upsample_bilinear(x, size=128)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers):
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x.clone(),), submodels=['network', 'proj'], name_prefix=f'layers[{i}].')
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position

    def __repr__(self):
        repr = f'{self.__class__.__name__}(dim_z={self.dim_z}, spectral_norm={self.spectral_norm})'
        return repr


class CoordConvSinAct_EqualLR(nn.Module):
    """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.coord_conv = mod_conv_fc.EqualConv2d(2, out_channels, kernel_size, stride, padding=padding)
        self.sin_act = SinAct()
        self.conv = mod_conv_fc.EqualConv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        pass

    def forward(self, input):
        batch, _, H, W = input.shape
        x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=input.device), torch.linspace(-1, 1, H, device=input.device))
        x = x.T
        y = y.T
        xy = torch.stack((x, y), dim=0)
        xy = xy.expand((batch, -1, -1, -1))
        xy_fea = self.coord_conv(xy)
        xy_fea = self.sin_act(xy_fea)
        out = self.conv(input)
        out = xy_fea + out
        return out


class ResidualCCBlockFirstDown_EqualLR(nn.Module):

    def __init__(self, inplanes, planes, stride=2, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip
        p = kernel_size // 2
        conv1 = CoordConvSinAct_EqualLR(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=p)
        conv2 = CoordConvSinAct_EqualLR(in_channels=planes, out_channels=planes, kernel_size=kernel_size, stride=1, padding=p)
        self.network = nn.Sequential(conv1, nn.LeakyReLU(0.2, inplace=True), conv2, nn.LeakyReLU(0.2, inplace=True))
        if skip:
            self.proj = mod_conv_fc.EqualConv2d(inplanes, planes, 1, stride)
        pass

    def forward(self, input):
        y = self.network(input)
        if self.skip:
            identity = self.proj(input)
            y = (y + identity) / math.sqrt(2)
        return y


class DiscriminatorMultiScale_EqualLR(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        """

    :param kwargs:
    """
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.dim_z = dim_z
        logger = logging.getLogger('tl')
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCCBlockFirstDown_EqualLR(3, 32), ResidualCCBlockFirstDown_EqualLR(32, 64), ResidualCCBlockFirstDown_EqualLR(64, 128), ResidualCCBlockFirstDown_EqualLR(128, 256), ResidualCCBlockFirstDown_EqualLR(256, 512), ResidualCCBlockFirstDown_EqualLR(512, 512), ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1), ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1), ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1)])
        final_layer = nn.Linear(512, 1 + dim_z + 2)
        self.final_layer = final_layer
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'final_layer': self.final_layer, 'D': self})
        logger.info(self)
        pass

    def forward(self, x, alpha, **kwargs):
        img_size = x.shape[-1]
        if img_size < 128:
            x = F.upsample_bilinear(x, size=128)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers):
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x.clone(),), submodels=['network', 'network.0', 'network.2'], name_prefix=f'layers[{i}].', input_padding=50)
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class ResidualCCBlock_FirstDown(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConv(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p), nn.LeakyReLU(0.2, inplace=True), CoordConv(planes, planes, kernel_size=kernel_size, stride=1, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class CCSEncoderDiscriminator_FirstDown(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.epoch = 0
        self.step = 0
        self.dim_z = dim_z
        self.layers = nn.ModuleList([ResidualCCBlock_FirstDown(32, 64), ResidualCCBlock_FirstDown(64, 128), ResidualCCBlock_FirstDown(128, 256), ResidualCCBlock_FirstDown(256, 400), ResidualCCBlock_FirstDown(400, 400), ResidualCCBlock_FirstDown(400, 400), ResidualCCBlock_FirstDown(400, 400)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1 + self.dim_z + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'fromRGB': self.fromRGB, 'final_layer': self.final_layer, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.fromRGB[start], inputs_args=(input,), submodels=['model'], name_prefix=f'fromRGB[{start}].')
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x,), submodels=['network', 'network.0', 'network.2'], name_prefix=f'layers[{start + i}].')
            x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class ResidualCCBlock_FirstDown_SinAct(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConv(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p), SinAct(), CoordConv(planes, planes, kernel_size=kernel_size, stride=1, padding=p), SinAct())
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class CCSEncoderDiscriminator_FirstDown_SinAct(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.epoch = 0
        self.step = 0
        self.dim_z = dim_z
        self.layers = nn.ModuleList([ResidualCCBlock_FirstDown_SinAct(32, 64), ResidualCCBlock_FirstDown_SinAct(64, 128), ResidualCCBlock_FirstDown_SinAct(128, 256), ResidualCCBlock_FirstDown_SinAct(256, 400), ResidualCCBlock_FirstDown_SinAct(400, 400), ResidualCCBlock_FirstDown_SinAct(400, 400), ResidualCCBlock_FirstDown_SinAct(400, 400)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1 + self.dim_z + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'fromRGB': self.fromRGB, 'final_layer': self.final_layer, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.fromRGB[start], inputs_args=(input,), submodels=['model'], name_prefix=f'fromRGB[{start}].')
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x,), submodels=['network', 'network.0', 'network.2'], name_prefix=f'layers[{start + i}].')
            x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class CoordConvSinAct(nn.Module):
    """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

    def __init__(self, in_channels, out_channels, channels_per_group=16, **kwargs):
        super().__init__()
        self.coord_conv = nn.Conv2d(2, out_channels, **kwargs)
        self.sin_act = SinAct()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        pass

    def forward(self, input):
        batch, _, H, W = input.shape
        x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=input.device), torch.linspace(-1, 1, H, device=input.device))
        x = x.T
        y = y.T
        xy = torch.stack((x, y), dim=0)
        xy = xy.expand((batch, -1, -1, -1))
        xy_fea = self.coord_conv(xy)
        xy_fea = self.sin_act(xy_fea)
        out = self.conv(input)
        out = xy_fea + out
        return out


class ResidualCCBlock_FirstDown_CoordConvSinAct(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConvSinAct(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p), nn.LeakyReLU(0.2, inplace=True), CoordConvSinAct(planes, planes, kernel_size=kernel_size, stride=1, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class CCSEncoderDiscriminator_FirstDown_CoordConvSinAct(nn.Module):
    """
  Coord_Conv_Sin (good);

  """

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.epoch = 0
        self.step = 0
        self.dim_z = dim_z
        max_channel = 400
        self.layers = nn.ModuleList([ResidualCCBlock_FirstDown_CoordConvSinAct(32, 64), ResidualCCBlock_FirstDown_CoordConvSinAct(64, 128), ResidualCCBlock_FirstDown_CoordConvSinAct(128, 256), ResidualCCBlock_FirstDown_CoordConvSinAct(256, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel)])
        self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'fromRGB': self.fromRGB, 'final_layer': self.final_layer, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.fromRGB[start], inputs_args=(input,), submodels=['model'], name_prefix=f'fromRGB[{start}].')
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x,), submodels=['network', 'network.0', 'network.2'], input_padding=50, name_prefix=f'layers[{start + i}].')
            x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class Discriminator_CoordConvSinAct(nn.Module):
    """
  Coord_Conv_Sin (good);
  Support 512 and 1024;

  """

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.epoch = 0
        self.step = 0
        self.dim_z = dim_z
        max_channel = 400
        self.layers = nn.ModuleList([ResidualCCBlock_FirstDown_CoordConvSinAct(32, 64), ResidualCCBlock_FirstDown_CoordConvSinAct(64, 128), ResidualCCBlock_FirstDown_CoordConvSinAct(128, 256), ResidualCCBlock_FirstDown_CoordConvSinAct(256, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel)])
        self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'fromRGB': self.fromRGB, 'final_layer': self.final_layer, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.fromRGB[start], inputs_args=(input,), submodels=['model'], name_prefix=f'fromRGB[{start}].')
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x,), submodels=['network', 'network.0', 'network.2'], input_padding=50, name_prefix=f'layers[{start + i}].')
            x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConvSinAct_EqualLR(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p), nn.LeakyReLU(0.2, inplace=True), CoordConvSinAct_EqualLR(planes, planes, kernel_size=kernel_size, stride=1, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = mod_conv_fc.EqualConv2d(inplanes, planes, 1, 2)
        pass

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class CCSEncoderDiscriminator_FirstDown_CoordConvSinAct_EqualLR(nn.Module):
    """
  CoordConv + GroupNorm
  """

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, dim_z=0, **kwargs):
        super().__init__()
        self.repr = f'dim_z={dim_z}'
        self.epoch = 0
        self.step = 0
        self.dim_z = dim_z
        max_channel = 400
        self.layers = nn.ModuleList([ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(32, 64), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(64, 128), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(128, 256), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(256, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel), ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel), AdapterBlock(max_channel)])
        self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        torch_utils.print_number_params(models_dict={'layers': self.layers, 'fromRGB': self.fromRGB, 'final_layer': self.final_layer, 'D': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.fromRGB[start], inputs_args=(input,), submodels=['model'], name_prefix=f'fromRGB[{start}].')
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x,), submodels=['network', 'network.0', 'network.2'], input_padding=50, name_prefix=f'layers[{start + i}].')
            x = layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix=f'final_layer.')
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:1 + self.dim_z]
        position = x[..., 1 + self.dim_z:]
        return prediction, latent, position


class INRNetwork_Skip_LinearSinMod(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, input_dim, style_dim, hidden_dim=256, pre_rgb_dim=32, device=None, name_prefix='inr', **kwargs):
        """

    :param input_dim:
    :param style_dim:
    :param hidden_dim:
    :param pre_rgb_dim:
    :param device:
    :param name_prefix:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'input_dim={input_dim}, style_dim={style_dim}, hidden_dim={hidden_dim}, pre_rgb_dim={pre_rgb_dim}'
        self.device = device
        self.pre_rgb_dim = pre_rgb_dim
        self.name_prefix = name_prefix
        self.channels = {'4': hidden_dim, '8': hidden_dim, '16': hidden_dim, '32': hidden_dim, '64': hidden_dim, '128': hidden_dim, '256': hidden_dim, '512': hidden_dim, '1024': hidden_dim}
        self.style_dim_dict = {}
        _out_dim = input_dim
        network = OrderedDict()
        to_rbgs = OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel
            if name.startswith(('none',)):
                _linear_block = inr_network.LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = SinBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block
            _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=pre_rgb_dim, use_equal_fc=False)
            to_rbgs[name] = _to_rgb
        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(inr_network.frequency_init(100))
        out_layers = []
        if pre_rgb_dim > 3:
            out_layers.append(nn.Linear(pre_rgb_dim, 3))
        out_layers.append(nn.Tanh())
        self.tanh = nn.Sequential(*out_layers)
        self.tanh.apply(inr_network.frequency_init(100))
        torch_utils.print_number_params({'network': self.network, 'to_rbgs': self.to_rgbs, 'tanh': self.tanh, 'inr_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward(self, input, style_dict, img_size=1024, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        img_size = str(2 ** int(np.log2(img_size)))
        rgb = 0
        for idx, (name, block) in enumerate(self.network.items()):
            if idx >= 4:
                skip = True
            else:
                skip = False
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(block, inputs_args=(x, style_dict, skip), submodels=['mod1', 'mod2'], name_prefix=f'block.{name}.')
            x = block(x, style_dict, skip=skip)
            if idx >= 3:
                if global_cfg.tl_debug:
                    VerboseModel.forward_verbose(self.to_rgbs[name], inputs_args=(x, rgb), name_prefix=f'to_rgb.{name}.')
                rgb = self.to_rgbs[name](x, skip=rgb)
            if name == img_size:
                break
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.tanh, inputs_args=(rgb,), name_prefix='tanh.')
        out = self.tanh(rgb)
        return out


class NeRFNetwork_xyz_d_posenc(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim=3, hidden_dim=256, hidden_layers=2, style_dim=512, rgb_dim=3, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, style_dim={style_dim}, rgb_dim={rgb_dim}, '
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.xyz_emb = comm_model_utils.PosEmbedding(max_logscale=9, N_freqs=10)
        dim_xyz_emb = self.xyz_emb.get_out_dim()
        self.dir_emb = comm_model_utils.PosEmbedding(max_logscale=3, N_freqs=4)
        dim_dir_emb = self.dir_emb.get_out_dim()
        self.style_dim_dict = {}
        _out_dim = dim_xyz_emb
        self.network = nn.ModuleList()
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            if True:
                _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            else:
                _layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim + dim_dir_emb
        _out_dim = hidden_dim // 2
        self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = nerf_network.UniformBoxWarp(0.24)
        torch_utils.print_number_params({'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        xyz_emb = self.xyz_emb(input)
        x = xyz_emb
        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
            x = layer(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        style = style_dict[f'{self.name_prefix}_rgb']
        dir_emb = self.dir_emb(ray_directions)
        x = torch.cat([dir_emb, x], dim=-1)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_sine, inputs_args=(x, style), name_prefix=f'color_layer_sine.')
        x = self.color_layer_sine(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        out = self.forward_with_frequencies_phase_shifts(input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs)
        return out

    def print_number_params(self):
        None
        pass

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(self, transformed_points, transformed_ray_directions_expanded, style_dict, max_points, num_steps):
        batch_size, num_points, _ = transformed_points.shape
        rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1), device=self.device)
        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b:b + 1, head:tail] = self(input=transformed_points[b:b + 1, head:tail], style_dict={name: style[b:b + 1] for name, style in style_dict.items()}, ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                head += max_points
        rgb_sigma_output = rearrange(rgb_sigma_output, 'b (hw s) rgb_sigma -> b hw s rgb_sigma', s=num_steps)
        return rgb_sigma_output


class NeRFNetwork_xyz_posenc(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim=3, hidden_dim=256, hidden_layers=2, style_dim=512, rgb_dim=3, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, style_dim={style_dim}, rgb_dim={rgb_dim}, '
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.xyz_emb = comm_model_utils.PosEmbedding(max_logscale=9, N_freqs=10)
        dim_xyz_emb = self.xyz_emb.get_out_dim()
        self.dir_emb = nn.Identity()
        dim_dir_emb = 0
        self.style_dim_dict = {}
        _out_dim = dim_xyz_emb
        self.network = nn.ModuleList()
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            if True:
                _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            else:
                _layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim + dim_dir_emb
        _out_dim = hidden_dim // 2
        self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = nerf_network.UniformBoxWarp(0.24)
        torch_utils.print_number_params({'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass

    def forward_with_frequencies_phase_shifts(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper)])), inputs_args=(input,), name_prefix='xyz.')
        input = self.gridwarper(input)
        xyz_emb = self.xyz_emb(input)
        x = xyz_emb
        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(layer, inputs_args=(x, style), name_prefix=f'network.{index}.')
            x = layer(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.final_layer, inputs_args=(x,), name_prefix='final_layer')
        sigma = self.final_layer(x)
        style = style_dict[f'{self.name_prefix}_rgb']
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_sine, inputs_args=(x, style), name_prefix=f'color_layer_sine.')
        x = self.color_layer_sine(x, style)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.color_layer_linear, inputs_args=(x,), name_prefix='color_layer_linear.')
        rbg = self.color_layer_linear(x)
        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        out = self.forward_with_frequencies_phase_shifts(input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs)
        return out

    def print_number_params(self):
        None
        pass

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(self, transformed_points, transformed_ray_directions_expanded, style_dict, max_points, num_steps):
        batch_size, num_points, _ = transformed_points.shape
        rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1), device=self.device)
        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b:b + 1, head:tail] = self(input=transformed_points[b:b + 1, head:tail], style_dict={name: style[b:b + 1] for name, style in style_dict.items()}, ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                head += max_points
        rgb_sigma_output = rearrange(rgb_sigma_output, 'b (hw s) rgb_sigma -> b hw s rgb_sigma', s=num_steps)
        return rgb_sigma_output


class NeRFNetwork_xyz_d_sine(NeRFNetwork_xyz_d_posenc):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, in_dim=3, hidden_dim=256, hidden_layers=2, style_dim=512, rgb_dim=3, device=None, name_prefix='nerf', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super(NeRFNetwork_xyz_d_posenc, self).__init__()
        self.repr = f'in_dim={in_dim}, hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, style_dim={style_dim}, rgb_dim={rgb_dim}, '
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix
        self.xyz_emb = nn.Identity()
        dim_xyz_emb = 3
        dim_dir_emb = 27
        self.dir_emb = nn.Sequential(nn.Linear(3, dim_dir_emb), SinAct())
        self.style_dim_dict = {}
        _out_dim = dim_xyz_emb
        self.network = nn.ModuleList()
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim
            if True:
                _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            else:
                _layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
        self.final_layer = nn.Linear(hidden_dim, 1)
        _in_dim = hidden_dim + dim_dir_emb
        _out_dim = hidden_dim // 2
        self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim))
        self.color_layer_linear.apply(init_func.kaiming_leaky_init)
        self.dim_styles = sum(self.style_dim_dict.values())
        self.gridwarper = nerf_network.UniformBoxWarp(0.24)
        torch_utils.print_number_params({'dir_emb': self.dir_emb, 'network': self.network, 'final_layer': self.final_layer, 'color_layer_sine': self.color_layer_sine, 'color_layer_linear': self.color_layer_linear, 'nerf_net': self})
        logging.getLogger('tl').info(self)
        pass


class INRNetwork_CLN(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_layers, rgb_dim=3, device=None, name_prefix='inr', **kwargs):
        """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix
        self.style_dim_dict = {}
        self.linear_layers = nn.ModuleList()
        self.cln_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        for i in range(hidden_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim
            linear_layer = nn.Linear(in_dim, out_dim)
            self.linear_layers.append(linear_layer)
            cln_layer = CLN(in_dim=out_dim)
            self.cln_layers.append(cln_layer)
            self.style_dim_dict[f'{name_prefix}_cln_{i}'] = cln_layer.style_dim
            act_layer = nn.LeakyReLU(0.2)
            self.act_layers.append(act_layer)
        self.to_rbg = nn.Sequential(nn.Linear(hidden_dim, rgb_dim), nn.Tanh())
        self.to_rbg.apply(frequency_init(25))
        self.dim_styles = sum(self.style_dim_dict.values())
        self.print_number_params()
        pass

    def forward(self, input, style_dict, **kwargs):
        """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
        x = input
        for index, (linear_layer, cln_layer, act_layer) in enumerate(zip(self.linear_layers, self.cln_layers, self.act_layers)):
            x = linear_layer(x)
            style = style_dict[f'{self.name_prefix}_cln_{index}']
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(cln_layer, inputs_args=(x, style), name_prefix=f'{self.name_prefix}.cln_layers.{index}.')
            x = cln_layer(x, style)
            x = act_layer(x)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.to_rbg, inputs_args=(x,), name_prefix='to_rgb.')
        out = self.to_rbg(x)
        return out

    def print_number_params(self):
        None
        torch_utils.print_number_params({'linear_layers': self.linear_layers, 'cln_layers': self.cln_layers, 'to_rbg': self.to_rbg, 'inr_net': self})
        pass


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30.0 * x)


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(input_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)
        return torch.cat([rbg, sigma], dim=-1)


def modified_first_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1), coordinates.reshape(batch_size, 1, 1, -1, n_dims), mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(32 + 3, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        None
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96) * 0.01)
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([rbg, sigma], dim=-1)


class EmbeddingPiGAN256(EmbeddingPiGAN128):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64) * 0.1)


class FiLMLayerEqualFC(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = EqualLinear(input_dim, hidden_dim)
        pass

    def forward(self, x, freq, phase_shift):
        """

    :param x: (b, num_points, d)
    :param freq: (b, d)
    :param phase_shift: (b, d)
    :return:
    """
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        out = torch.sin(freq * x + phase_shift)
        return out


class MultiHead_MappingNetwork_EqualFC(nn.Module):

    def __init__(self, z_dim, hidden_dim, base_layers, head_layers, head_dim_dict):
        super().__init__()
        self.head_dim_dict = head_dim_dict
        self.norm_layer = PixelNorm()
        base_net = []
        for i in range(base_layers):
            if i == 0:
                in_dim = z_dim
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim
            if i != base_layers - 1:
                act = 'fused_leaky_relu'
            else:
                act = None
            hidden_layer = EqualLinear(in_dim=in_dim, out_dim=out_dim, bias=True, lr_mul=0.01, activation=act)
            base_net.append(hidden_layer)
        self.base_net = nn.Sequential(*base_net)
        for name, head_dim in head_dim_dict.items():
            head_net = []
            in_dim = hidden_dim
            for i in range(head_layers):
                if i == head_layers - 1:
                    out_dim = head_dim
                    act = None
                else:
                    out_dim = hidden_dim
                    act = 'fused_leaky_relu'
                hidden_layer = EqualLinear(in_dim=in_dim, out_dim=out_dim, bias=True, lr_mul=0.01, activation=act)
                head_net.append(hidden_layer)
            self.add_module(name, nn.Sequential(*head_net))
        self.print_number_params()
        pass

    def forward(self, z):
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.norm_layer, inputs_args=(z,), name_prefix='norm_layer')
        z = self.norm_layer(z)
        if global_cfg.tl_debug:
            VerboseModel.forward_verbose(self.base_net, inputs_args=(z,), name_prefix='base_net.')
        base_fea = self.base_net(z)
        out_dict = {}
        for name, head_dim in self.head_dim_dict.items():
            head_net = getattr(self, name)
            if global_cfg.tl_debug:
                VerboseModel.forward_verbose(head_net, inputs_args=(base_fea,), name_prefix=f'{name}.')
            out = head_net(base_fea)
            out_dict[name] = out
        return out_dict

    def print_number_params(self):
        models_dict = {'base_net': self.base_net}
        for name, _ in self.head_dim_dict.items():
            models_dict[name] = getattr(self, name)
        models_dict['mapping_network'] = self
        None
        torch_utils.print_number_params(models_dict)
        pass


class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([2, 3])


class ResidualCoordConvBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(CoordConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p), nn.LeakyReLU(0.2, inplace=True), CoordConv(planes, planes, kernel_size=kernel_size, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample

    def forward(self, identity):
        y = self.network(identity)
        if self.downsample:
            y = nn.functional.avg_pool2d(y, 2)
        if self.downsample:
            identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)
        y = (y + identity) / math.sqrt(2)
        return y


class ProgressiveDiscriminator(nn.Module):
    """Implement of a progressive growing discriminator with ResidualCoordConv Blocks"""

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCoordConvBlock(16, 32, downsample=True), ResidualCoordConvBlock(32, 64, downsample=True), ResidualCoordConvBlock(64, 128, downsample=True), ResidualCoordConvBlock(128, 256, downsample=True), ResidualCoordConvBlock(256, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True)])
        self.fromRGB = nn.ModuleList([AdapterBlock(16), AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {(2): 8, (4): 7, (8): 6, (16): 5, (32): 4, (64): 3, (128): 2, (256): 1, (512): 0}

    def forward(self, input, alpha, instance_noise=0, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], 1)
        return x


class ProgressiveEncoderDiscriminator(nn.Module):
    """
    Implement of a progressive growing discriminator with ResidualCoordConv Blocks.
    Identical to ProgressiveDiscriminator except it also predicts camera angles and latent codes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCoordConvBlock(16, 32, downsample=True), ResidualCoordConvBlock(32, 64, downsample=True), ResidualCoordConvBlock(64, 128, downsample=True), ResidualCoordConvBlock(128, 256, downsample=True), ResidualCoordConvBlock(256, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True), ResidualCoordConvBlock(400, 400, downsample=True)])
        self.fromRGB = nn.ModuleList([AdapterBlock(16), AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2)
        self.img_size_to_layer = {(2): 8, (4): 7, (8): 6, (16): 5, (32): 4, (64): 3, (128): 2, (256): 1, (512): 0}

    def forward(self, input, alpha, instance_noise=0, **kwargs):
        if instance_noise > 0:
            input = input + torch.randn_like(input) * instance_noise
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], -1)
        prediction = x[..., 0:1]
        latent = x[..., 1:257]
        position = x[..., 257:259]
        return prediction, latent, position


class StridedResidualConvBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=p), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=2, padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity) / math.sqrt(2)
        return y


class StridedDiscriminator(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([StridedResidualConvBlock(32, 64), StridedResidualConvBlock(64, 128), StridedResidualConvBlock(128, 256), StridedResidualConvBlock(256, 400), StridedResidualConvBlock(400, 400), StridedResidualConvBlock(400, 400), StridedResidualConvBlock(400, 400)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        self.pose_layer = nn.Linear(2, 400)

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], 1)
        return x, None, None


class CCSDiscriminator(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([ResidualCCBlock(32, 64), ResidualCCBlock(64, 128), ResidualCCBlock(128, 256), ResidualCCBlock(256, 400), ResidualCCBlock(400, 400), ResidualCCBlock(400, 400), ResidualCCBlock(400, 400)])
        self.fromRGB = nn.ModuleList([AdapterBlock(32), AdapterBlock(64), AdapterBlock(128), AdapterBlock(256), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400), AdapterBlock(400)])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {(2): 7, (4): 6, (8): 5, (16): 4, (32): 3, (64): 2, (128): 1, (256): 0}
        self.pose_layer = nn.Linear(2, 400)

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)
        x = self.final_layer(x).reshape(x.shape[0], 1)
        return x, None, None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdapterBlock,
     lambda: ([], {'output_channels': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AddLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CCSDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (CCSEncoderDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (CLNLayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConvSinAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConvSinAct_EqualLR,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordFC,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CustomMappingNetwork,
     lambda: ([], {'z_dim': 4, 'map_hidden_dim': 4, 'map_output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualConvTranspose2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FiLMLayer,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (FiLMLayerEqualFC,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FiLMLayer_PreSin,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GlobalAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearScale,
     lambda: ([], {'scale': 1.0, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearSinAct,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModToRGB,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PosEmbedding,
     lambda: ([], {'max_logscale': 1.0, 'N_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProgressiveDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 256, 32, 32])], {}),
     False),
    (ProgressiveEncoderDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 256, 32, 32])], {}),
     False),
    (ResidualCCBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCCBlockFirstDown,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualCCBlockFirstDown_EqualLR,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCCBlock_FirstDown,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCCBlock_FirstDown_CoordConvSinAct,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCCBlock_FirstDown_SinAct,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualCoordConvBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SinAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SinActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (StridedDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (StridedResidualConvBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ToRGB,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UniformBoxWarp,
     lambda: ([], {'sidelength': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_PeterouZh_CIPS_3D(_paritybench_base):
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

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

