import sys
_module = sys.modules[__name__]
del sys
DIP = _module
denoising = _module
inpainting = _module
models = _module
common = _module
common_test = _module
cross_skip = _module
downsampler = _module
gen_upsample_layer = _module
model_denoising = _module
model_inpainting = _module
model_sr = _module
ref = _module
resnet = _module
skip = _module
skip_search_up = _module
texture_nets = _module
unet = _module
unet_search_up = _module
utils = _module
common_utils = _module
denoising_utils = _module
feature_inversion_utils = _module
inpainting_utils = _module
load_image = _module
matcher = _module
perceptual_loss = _module
matcher = _module
perceptual_loss = _module
vgg_modified = _module
sr_utils = _module
timer = _module
NAS = _module
demo = _module
gen_id = _module
gen_upsample_layer = _module
genotypes = _module
model = _module
model_gen = _module
operations = _module
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


import matplotlib


import matplotlib.pyplot as plt


import random


import numpy as np


import torch


import torch.optim


import warnings


import torch.nn as nn


from numpy.random import normal


from numpy.linalg import svd


from math import sqrt


import torch.nn.init


import torch.nn.functional as F


import torchvision


import torchvision.transforms as transforms


import torchvision.models as models


from collections import OrderedDict


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision.models.vgg import model_urls


from torchvision.models import vgg19


from torch.autograd import Variable


class Concat(nn.Module):

    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2, diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):

    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        b = torch.zeros(a).type_as(input.data)
        b.normal_()
        x = torch.autograd.Variable(b)
        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1.0 / (kernel_width * kernel_width)
    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        center = (kernel_width + 1.0) / 2.0
        None
        sigma_sq = sigma * sigma
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.0
                dj = (j - center) / 2.0
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2.0 * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.0
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                pi_sq = np.pi * np.pi
                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                kernel[i - 1][j - 1] = val
    else:
        assert False, 'wrong method name'
    kernel /= kernel.sum()
    return kernel


class Downsampler(nn.Module):
    """
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'
        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'
        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1.0 / np.sqrt(2)
            kernel_type_ = 'gauss'
        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type
        else:
            assert False, 'wrong name kernel'
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0
        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch
        self.downsampler_ = downsampler
        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.0)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.0)
            self.padding = nn.ReplicationPad2d(pad)
        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False
        stride = 1
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


class OutputBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias, pad, need_sigmoid):
        super(OutputBlock, self).__init__()
        if need_sigmoid:
            self.op = nn.Sequential(conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad), nn.Sigmoid())
        else:
            self.op = nn.Sequential(conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad))

    def forward(self, data):
        return self.op(data)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, model_index):
        super(UpsampleBlock, self).__init__()
        self.op = gen_upsample_layer.gen_layer(C_in=in_channel, C_out=out_channel, model_index=model_index)

    def forward(self, data):
        return self.op(data)


def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


class DownsampleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias, pad, act_fun, downsample_mode):
        super(DownsampleBlock, self).__init__()
        self.op = nn.Sequential(conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=2, bias=bias, pad=pad, downsample_mode=downsample_mode), bn(num_features=out_channel), act(act_fun=act_fun))

    def forward(self, data):
        return self.op(data)


class SkipBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias, pad, act_fun):
        super(SkipBlock, self).__init__()
        self.op = nn.Sequential(conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun))

    def forward(self, data):
        return self.op(data)


class EncoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias, pad, act_fun, downsample_mode):
        super(EncoderBlock, self).__init__()
        self.op = nn.Sequential(conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=2, bias=bias, pad=pad, downsample_mode=downsample_mode), bn(num_features=out_channel), act(act_fun=act_fun), conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun))

    def forward(self, data):
        return self.op(data)


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bias, pad, act_fun, need1x1_up):
        super(DecoderBlock, self).__init__()
        if need1x1_up:
            self.op = nn.Sequential(bn(num_features=out_channel), conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=1, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun), conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun), conv(in_f=out_channel, out_f=out_channel, kernel_size=1, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun))
        else:
            self.op = nn.Sequential(bn(num_features=out_channel), conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=1, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun), conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad), bn(num_features=out_channel), act(act_fun=act_fun))

    def forward(self, data):
        return self.op(data)


def skip(model_index, num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True, pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    last_scale = n_scales - 1
    cur_depth = None
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper_main = nn.Sequential()
        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        C_in = num_channels_down[i]
        C_out = num_channels_down[i]
        deeper.add(gen_upsample_layer.gen_layer(C_in=C_in, C_out=C_out, model_index=model_index))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))
        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model


class Model(nn.Module):

    def __init__(self, model_index=119, num_input_channels=32, num_output_channels=3, num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True, pad='reflection', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
        super(Model, self).__init__()
        self.enc1 = EncoderBlock(in_channel=num_input_channels, out_channel=num_channels_down[0], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.enc2 = EncoderBlock(in_channel=num_channels_down[0], out_channel=num_channels_down[1], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.enc3 = EncoderBlock(in_channel=num_channels_down[1], out_channel=num_channels_down[2], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.enc4 = EncoderBlock(in_channel=num_channels_down[2], out_channel=num_channels_down[3], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.enc5 = EncoderBlock(in_channel=num_channels_down[3], out_channel=num_channels_down[4], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.skip1 = SkipBlock(in_channel=num_input_channels, out_channel=num_channels_up[0], kernel_size=1, bias=need_bias, pad=pad, act_fun=act_fun)
        self.skip2 = SkipBlock(in_channel=num_channels_down[0], out_channel=num_channels_up[1], kernel_size=1, bias=need_bias, pad=pad, act_fun=act_fun)
        self.skip3 = SkipBlock(in_channel=num_channels_down[1], out_channel=num_channels_up[2], kernel_size=1, bias=need_bias, pad=pad, act_fun=act_fun)
        self.skip4 = SkipBlock(in_channel=num_channels_down[2], out_channel=num_channels_up[3], kernel_size=1, bias=need_bias, pad=pad, act_fun=act_fun)
        self.skip5 = SkipBlock(in_channel=num_channels_down[3], out_channel=num_channels_up[4], kernel_size=1, bias=need_bias, pad=pad, act_fun=act_fun)
        self.skip_up_5_4 = UpsampleBlock(in_channel=num_channels_down[4], out_channel=num_channels_up[3], model_index=model_index)
        self.skip_up_4_3 = UpsampleBlock(in_channel=num_channels_down[3], out_channel=num_channels_up[2], model_index=model_index)
        self.skip_up_3_2 = UpsampleBlock(in_channel=num_channels_down[2], out_channel=num_channels_up[1], model_index=model_index)
        self.skip_up_2_1 = UpsampleBlock(in_channel=num_channels_down[1], out_channel=num_channels_up[0], model_index=model_index)
        self.skip_down_1_2 = DownsampleBlock(in_channel=num_input_channels, out_channel=num_channels_up[0], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.skip_down_2_3 = DownsampleBlock(in_channel=num_channels_down[0], out_channel=num_channels_up[1], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.skip_down_3_4 = DownsampleBlock(in_channel=num_channels_down[1], out_channel=num_channels_up[2], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.skip_down_4_5 = DownsampleBlock(in_channel=num_channels_down[2], out_channel=num_channels_up[3], kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)
        self.up5 = UpsampleBlock(in_channel=num_channels_up[4], out_channel=num_channels_up[4], model_index=model_index)
        self.up4 = UpsampleBlock(in_channel=num_channels_up[3], out_channel=num_channels_up[3], model_index=model_index)
        self.up3 = UpsampleBlock(in_channel=num_channels_up[2], out_channel=num_channels_up[2], model_index=model_index)
        self.up2 = UpsampleBlock(in_channel=num_channels_up[1], out_channel=num_channels_up[1], model_index=model_index)
        self.up1 = UpsampleBlock(in_channel=num_channels_up[0], out_channel=num_channels_up[0], model_index=model_index)
        self.dec5 = DecoderBlock(in_channel=num_channels_down[4], out_channel=num_channels_up[4], kernel_size=filter_size_up, bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.dec4 = DecoderBlock(in_channel=num_channels_up[3], out_channel=num_channels_up[3], kernel_size=filter_size_up, bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.dec3 = DecoderBlock(in_channel=num_channels_up[2], out_channel=num_channels_up[2], kernel_size=filter_size_up, bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.dec2 = DecoderBlock(in_channel=num_channels_up[1], out_channel=num_channels_up[1], kernel_size=filter_size_up, bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.dec1 = DecoderBlock(in_channel=num_channels_up[0], out_channel=num_channels_up[0], kernel_size=filter_size_up, bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.output = OutputBlock(in_channel=num_channels_up[0], out_channel=num_output_channels, kernel_size=1, bias=need_bias, pad=pad, need_sigmoid=need_sigmoid)

    def forward(self, data):
        enc1 = self.enc1(data)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        add5 = self.up5(enc5) + self.skip_down_4_5(enc3) + self.skip5(enc4)
        dec5 = self.dec5(add5)
        add4 = self.up4(dec5) + self.skip_down_3_4(enc2) + self.skip4(enc3)
        dec4 = self.dec4(add4)
        add3 = self.up3(dec4) + self.skip_down_2_3(enc1) + self.skip3(enc2)
        dec3 = self.dec3(add3)
        add2 = self.up2(dec3) + self.skip_down_1_2(data) + self.skip2(enc1) + self.skip_up_3_2(self.skip_up_4_3(self.skip_up_5_4(enc4)))
        dec2 = self.dec2(add2)
        add1 = self.up1(dec2) + self.skip1(data) + self.skip_up_2_1(self.skip_up_3_2(self.skip_up_4_3(enc3)))
        dec1 = self.dec1(add1)
        out = self.output(dec1)
        return out


class ResidualSequential(nn.Sequential):

    def __init__(self, *args):
        super(ResidualSequential, self).__init__(*args)

    def forward(self, x):
        out = super(ResidualSequential, self).forward(x)
        x_ = None
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            diff2 = x.size(2) - out.size(2)
            diff3 = x.size(3) - out.size(3)
            x_ = x[:, :, diff2 / 2:out.size(2) + diff2 / 2, diff3 / 2:out.size(3) + diff3 / 2]
        else:
            x_ = x
        return out + x_

    def eval(self):
        None
        for m in self.modules():
            m.eval()
        exit()


def get_block(num_channels, norm_layer, act_fun):
    layers = [nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False), norm_layer(num_channels, affine=True), act(act_fun), nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False), norm_layer(num_channels, affine=True)]
    return layers


class ResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, num_blocks, num_channels, need_residual=True, act_fun='LeakyReLU', need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='reflection'):
        """
            pad = 'start|zero|replication'
        """
        super(ResNet, self).__init__()
        if need_residual:
            s = ResidualSequential
        else:
            s = nn.Sequential
        stride = 1
        layers = [conv(num_input_channels, num_channels, 3, stride=1, bias=True, pad=pad), act(act_fun)]
        for i in range(num_blocks):
            layers += [s(*get_block(num_channels, norm_layer, act_fun))]
        layers += [nn.Conv2d(num_channels, num_channels, 3, 1, 1), norm_layer(num_channels, affine=True)]
        layers += [conv(num_channels, num_output_channels, 3, 1, bias=True, pad=pad), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

    def eval(self):
        self.model.eval()


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad), norm_layer(out_size), nn.ReLU())
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad), norm_layer(out_size), nn.ReLU())
        else:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad), nn.ReLU())
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):

    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class unetUp(nn.Module):

    def __init__(self, out_size, need_bias, pad, model_index, use_act, same_num_filt=False):
        super(unetUp, self).__init__()
        num_filt = out_size if same_num_filt else out_size * 2
        self.up = gen_upsample_layer.gen_layer(C_in=num_filt, C_out=out_size, use_act=use_act, model_index=model_index)
        self.conv = unetConv2(out_size * 2, out_size, None, need_bias, pad)

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        if inputs2.size(2) != in1_up.size(2) or inputs2.size(3) != in1_up.size(3):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2:diff2 + in1_up.size(2), diff3:diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2
        output = self.conv(torch.cat([in1_up, inputs2_], 1))
        return output


class UNet(nn.Module):
    """
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    """

    def __init__(self, model_index, use_act, num_input_channels=3, num_output_channels=3, feature_scale=4, more_layers=0, concat_x=False, upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet, self).__init__()
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x
        filters = [64, 128, 256, 512, 1024]
        filters = [(x // self.feature_scale) for x in filters]
        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)
        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad)
        if self.more_layers > 0:
            self.more_downs = [unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], need_bias, pad, same_num_filt=True, model_index=model_index, use_act=use_act) for i in range(self.more_layers)]
            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)
        self.up4 = unetUp(filters[3], need_bias, pad, model_index=model_index, use_act=use_act)
        self.up3 = unetUp(filters[2], need_bias, pad, model_index=model_index, use_act=use_act)
        self.up2 = unetUp(filters[1], need_bias, pad, model_index=model_index, use_act=use_act)
        self.up1 = unetUp(filters[0], need_bias, pad, model_index=model_index, use_act=use_act)
        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))
        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)
        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)
        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)
        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)
        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)
        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 5]], 1)
                prevs.append(out)
            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_ = l(up_, prevs[self.more - idx - 2])
        else:
            up_ = down4
        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)
        return self.final(up1)


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def features(x):
    return x


def gram_matrix(x):
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


all_features = {'gram_matrix': gram_matrix, 'features': features}


all_losses = {'mse': nn.MSELoss(), 'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss()}


class Matcher:

    def __init__(self, how='gram_matrix', loss='mse', map_index=933):
        self.mode = 'store'
        self.stored = {}
        self.losses = {}
        if how in all_features.keys():
            self.get_statistics = all_features[how]
        else:
            assert False
        pass
        if loss in all_losses.keys():
            self.loss = all_losses[loss]
        else:
            assert False
        self.map_index = map_index
        self.method = 'match'

    def __call__(self, module, features):
        statistics = self.get_statistics(features)
        self.statistics = statistics
        if self.mode == 'store':
            self.stored[module] = statistics.detach()
        elif self.mode == 'match':
            if statistics.ndimension() == 2:
                if self.method == 'maximize':
                    self.losses[module] = -statistics[0, self.map_index]
                else:
                    self.losses[module] = torch.abs(300 - statistics[0, self.map_index])
            else:
                ws = self.window_size
                t = statistics.detach() * 0
                s_cc = statistics[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws]
                t_cc = t[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws]
                t_cc[:, self.map_index, ...] = 1
                if self.method == 'maximize':
                    self.losses[module] = -(s_cc * t_cc.contiguous()).sum()
                else:
                    self.losses[module] = torch.abs(200 - s_cc * t_cc.contiguous()).sum()

    def clean(self):
        self.losses = {}


def get_matcher(vgg, opt):
    matcher = Matcher(opt['what'], 'mse', opt['map_idx'])

    def hook(module, input, output):
        matcher(module, output)
    for layer_name in opt['layers']:
        vgg._modules[layer_name].register_forward_hook(hook)
    return matcher


class VGGModified(nn.Module):

    def __init__(self, vgg19_orig, slope=0.01):
        super(VGGModified, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module(str(0), vgg19_orig.features[0])
        self.features.add_module(str(1), nn.LeakyReLU(slope, True))
        self.features.add_module(str(2), vgg19_orig.features[2])
        self.features.add_module(str(3), nn.LeakyReLU(slope, True))
        self.features.add_module(str(4), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(5), vgg19_orig.features[5])
        self.features.add_module(str(6), nn.LeakyReLU(slope, True))
        self.features.add_module(str(7), vgg19_orig.features[7])
        self.features.add_module(str(8), nn.LeakyReLU(slope, True))
        self.features.add_module(str(9), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(10), vgg19_orig.features[10])
        self.features.add_module(str(11), nn.LeakyReLU(slope, True))
        self.features.add_module(str(12), vgg19_orig.features[12])
        self.features.add_module(str(13), nn.LeakyReLU(slope, True))
        self.features.add_module(str(14), vgg19_orig.features[14])
        self.features.add_module(str(15), nn.LeakyReLU(slope, True))
        self.features.add_module(str(16), vgg19_orig.features[16])
        self.features.add_module(str(17), nn.LeakyReLU(slope, True))
        self.features.add_module(str(18), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(19), vgg19_orig.features[19])
        self.features.add_module(str(20), nn.LeakyReLU(slope, True))
        self.features.add_module(str(21), vgg19_orig.features[21])
        self.features.add_module(str(22), nn.LeakyReLU(slope, True))
        self.features.add_module(str(23), vgg19_orig.features[23])
        self.features.add_module(str(24), nn.LeakyReLU(slope, True))
        self.features.add_module(str(25), vgg19_orig.features[25])
        self.features.add_module(str(26), nn.LeakyReLU(slope, True))
        self.features.add_module(str(27), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(28), vgg19_orig.features[28])
        self.features.add_module(str(29), nn.LeakyReLU(slope, True))
        self.features.add_module(str(30), vgg19_orig.features[30])
        self.features.add_module(str(31), nn.LeakyReLU(slope, True))
        self.features.add_module(str(32), vgg19_orig.features[32])
        self.features.add_module(str(33), nn.LeakyReLU(slope, True))
        self.features.add_module(str(34), vgg19_orig.features[34])
        self.features.add_module(str(35), nn.LeakyReLU(slope, True))
        self.features.add_module(str(36), nn.AvgPool2d((2, 2), (2, 2)))
        self.classifier = nn.Sequential()
        self.classifier.add_module(str(0), vgg19_orig.classifier[0])
        self.classifier.add_module(str(1), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(2), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(3), vgg19_orig.classifier[3])
        self.classifier.add_module(str(4), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(5), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(6), vgg19_orig.classifier[6])

    def forward(self, x):
        return self.classifier(self.features.forward(x))


def get_vgg16_caffe():
    vgg = torch.load('vgg16-caffe-py3.pth')
    names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5', 'torch_view', 'fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'fc8']
    model = nn.Sequential()
    for n, m in zip(names, list(vgg)):
        model.add_module(n, m)
    return model


def get_vgg19_caffe():
    model = vgg19()
    model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
    vgg = model.features
    vgg_classifier = model.classifier
    names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'torch_view', 'fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8']
    model = nn.Sequential()
    for n, m in zip(names, list(vgg) + list(vgg_classifier)):
        model.add_module(n, m)
    model.load_state_dict(torch.load('vgg19-caffe-py3.pth'))
    return model


def get_pretrained_net(name):
    """Loads pretrained network"""
    if name == 'alexnet_caffe':
        if not os.path.exists('alexnet-torch_py3.pth'):
            None
            os.system('wget -O alexnet-torch_py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/77xSWvrDN0CiQtK/download')
        return torch.load('alexnet-torch_py3.pth')
    elif name == 'vgg19_caffe':
        if not os.path.exists('vgg19-caffe-py3.pth'):
            None
            os.system('wget -O vgg19-caffe-py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/HPcOFQTjXxbmp4X/download')
        vgg = get_vgg19_caffe()
        return vgg
    elif name == 'vgg16_caffe':
        if not os.path.exists('vgg16-caffe-py3.pth'):
            None
            os.system('wget -O vgg16-caffe-py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/TUZ62HnPKWdxyLr/download')
        vgg = get_vgg16_caffe()
        return vgg
    elif name == 'vgg19_pytorch_modified':
        model = VGGModified(vgg19(pretrained=False), 0.2)
        model.load_state_dict(torch.load('vgg_pytorch_modified.pkl')['state_dict'])
        return model
    else:
        assert False


vgg_mean = torch.FloatTensor([103.939, 116.779, 123.68]).view(3, 1, 1)


def vgg_preprocess_caffe(var):
    r, g, b = torch.chunk(var, 3, dim=1)
    bgr = torch.cat((b, g, r), 1)
    out = bgr * 255 - torch.autograd.Variable(vgg_mean).type(var.type())
    return out


mean_pytorch = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1))


std_pytorch = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1))


def vgg_preprocess_pytorch(var):
    return (var - mean_pytorch.type_as(var)) / std_pytorch.type_as(var)


class PerceputalLoss(nn.modules.loss._Loss):
    """ 
        Assumes input image is in range [0,1] if `input_range` is 'sigmoid', [-1, 1] if 'tanh' 
    """

    def __init__(self, input_range='sigmoid', net_type='vgg_torch', input_preprocessing='corresponding', match=[{'layers': [11, 20, 29], 'what': 'features'}]):
        if input_range not in ['sigmoid', 'tanh']:
            assert False
        self.net = get_pretrained_net(net_type)
        self.matchers = [get_matcher(self.net, match_opts) for match_opts in match]
        preprocessing_correspondence = {'vgg19_torch': vgg_preprocess_caffe, 'vgg16_torch': vgg_preprocess_caffe, 'vgg19_pytorch': vgg_preprocess_pytorch, 'vgg19_pytorch_modified': vgg_preprocess_pytorch}
        if input_preprocessing == 'corresponding':
            self.preprocess_input = preprocessing_correspondence[net_type]
        else:
            self.preprocessing = preprocessing_correspondence[input_preprocessing]

    def preprocess_input(self, x):
        if self.input_range == 'tanh':
            x = (x + 1.0) / 2.0
        return self.preprocess(x)

    def __call__(self, x, y):
        self.matcher_content.mode = 'store'
        self.net(self.preprocess_input(y))
        self.matcher_content.mode = 'match'
        self.net(self.preprocess_input(x))
        return sum([sum(matcher.losses.values()) for matcher in self.matchers])


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.LeakyReLU(0.2, inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):

    def __init__(self, genotype, C_prev, C_curr, C_prev_prev=None, op_type='downsample'):
        super(Cell, self).__init__()
        self.op_type = op_type
        if self.op_type == 'downsample':
            self.preprocess0 = operations.ReLUConvBN(C_prev, C_curr, 1, 1, 0)
            conv_op_names, indices = zip(*genotype.downsample_conv)
            op_names, _ = zip(*genotype.downsample_method)
            concat = genotype.downsample_concat
        else:
            self.preprocess0 = operations.ReLUConvBN(C_prev, C_curr, 1, 1, 0)
            if C_prev_prev is not None:
                self.preprocess1 = operations.ReLUConvBN(C_prev_prev, C_curr, 1, 1, 0)
            conv_op_names, indices = zip(*genotype.upsample_conv)
            op_names, _ = zip(*genotype.upsample_method)
            concat = genotype.upsample_concat
        self._compile(C_curr=C_curr, conv_op_names=conv_op_names, op_names=op_names, indices=indices, concat=concat)

    def _compile(self, C_curr, conv_op_names, op_names, indices, concat):
        assert len(op_names) == len(indices)
        assert len(conv_op_names) == len(indices)
        self.num_of_nodes = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for index in range(len(op_names)):
            if self.op_type == 'downsample':
                downsample_name = op_names[index]
                conv_name = conv_op_names[index]
                stride = 2 if indices[index] < 2 else 1
                downsample_op = operations.DOWNSAMPLE_OPS[downsample_name](C_in=C_curr, stride=stride)
                conv_op = operations.CONV_OPS[conv_name](C_in=C_curr, C_out=C_curr, affine=True)
                op = nn.Sequential(downsample_op, conv_op)
            else:
                upsample_name = op_names[index]
                conv_name = conv_op_names[index]
                stride = 2 if indices[index] < 2 else 1
                upsample_op = operations.UPSAMPLE_OPS[upsample_name](C_in=C_curr, stride=stride)
                conv_op = operations.CONV_OPS[conv_name](C_in=C_curr, C_out=C_curr, affine=True)
                op = nn.Sequential(upsample_op, conv_op)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, drop_prob, s1=None):
        s0 = self.preprocess0(s0)
        if s1 is None:
            s1 = s0
        else:
            s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self.num_of_nodes):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, operations.Identity):
                    h1 = utils.drop_path(h1, drop_prob)
                if not isinstance(op2, operations.Identity):
                    h2 = utils.drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        out = torch.cat([states[i] for i in self._concat], dim=1)
        return out


class Stem(nn.Module):

    def __init__(self, num_input_channel, num_output_channel, norm_layer, need_bias, pad):
        super(Stem, self).__init__()
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(num_input_channel, num_output_channel, 3, bias=need_bias, pad=pad), norm_layer(num_output_channel), nn.ReLU())
            self.conv2 = nn.Sequential(conv(num_output_channel, num_output_channel, 3, bias=need_bias, pad=pad), norm_layer(num_output_channel), nn.ReLU())
        else:
            self.conv1 = nn.Sequential(conv(num_input_channel, num_output_channel, 3, bias=need_bias, pad=pad), nn.ReLU())
            self.conv2 = nn.Sequential(conv(num_output_channel, num_output_channel, 3, bias=need_bias, pad=pad), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class NetworkDIP(nn.Module):

    def __init__(self, genotype, num_input_channel=3, num_output_channel=3, concat_x=False, need_bias=True, norm_layer=nn.InstanceNorm2d, pad='zero', filters=[64, 128, 256, 512, 1024], init_filters=3, feature_scale=4, drop_path_prob=0.2):
        super(NetworkDIP, self).__init__()
        self._layers = len(filters)
        self.drop_path_prob = drop_path_prob
        filters = [(x // feature_scale) for x in filters]
        stem_output_channel = filters[0] if not concat_x else filters[0] - num_input_channel
        self.stem = Stem(num_input_channel=num_input_channel, num_output_channel=stem_output_channel, norm_layer=norm_layer, need_bias=need_bias, pad=pad)
        self.cells = nn.ModuleList()
        """ Initializa downsample cells first """
        op_type = 'downsample'
        C_prev = stem_output_channel
        for i in range(self._layers):
            C_curr = filters[i]
            cell = Cell(genotype, C_prev=C_prev, C_curr=C_curr, op_type=op_type)
            self.cells += [cell]
            C_prev = cell.multiplier * C_curr
        """ Initializa upsample cells first """
        op_type = 'upsample'
        up_mode = genotype.upsample_method
        C_prev_prev = None
        for i in range(self._layers - 1, -1, -1):
            C_prev = cell.multiplier * filters[i]
            if i > 0:
                C_curr = filters[i - 1]
            else:
                C_curr = C_prev // 2
            cell = Cell(genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C_curr=C_curr, op_type=op_type)
            self.cells += [cell]
            C_prev_prev = self.cells[i - 1].multiplier * filters[i - 1]
        C_curr = cell.multiplier * C_curr
        self.last_layer = nn.Conv2d(in_channels=C_curr, out_channels=num_output_channel, kernel_size=1)
        self.last_activ = nn.Sigmoid()

    def forward(self, data):
        s0 = self.stem(data)
        output_list = []
        for i, cell in enumerate(self.cells):
            if i < len(self.cells) / 2 + 1:
                s0 = cell(s0, drop_prob=0)
                output_list.append(s0)
            else:
                s1 = output_list[len(self.cells) - 1 - i]
                s0 = cell(s0=s0, s1=s1, drop_prob=0)
        s0 = self.last_layer(s0)
        s0 = self.last_activ(s0)
        return s0


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        out = self.op(x)
        return out


class ConvDownSample(nn.Module):

    def __init__(self, C_in, kernel_size, stride, padding, dilation):
        super(ConvDownSample, self).__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, padding, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class BilinearConv(nn.Module):

    def __init__(self, C_in, kernel_size, stride, padding):
        super(BilinearConv, self).__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.Upsample(scale_factor=stride, mode='bilinear'), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, bias=False), nn.LeakyReLU(0.2, inplace=False))

    def forward(self, x):
        return self.op(x)


class TransConv(nn.Module):

    def __init__(self, C_in, kernel_size, stride, padding, affine=True):
        super(TransConv, self).__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.ConvTranspose2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride - 1), nn.LeakyReLU(0.2, inplace=False))

    def forward(self, x):
        return self.op(x)


class BilinearAdditive(nn.Module):

    def __init__(self, C_in, kernel_size, stride, padding, split=4):
        super(BilinearAdditive, self).__init__()
        self.chuck_size = int(C_in / split)
        self.op1 = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.Upsample(scale_factor=stride, mode='bilinear'))
        self.op2 = nn.Sequential(nn.Conv2d(int(C_in / split), C_in, kernel_size=kernel_size, padding=padding, bias=False), nn.LeakyReLU(0.2, inplace=False))

    def forward(self, x):
        out = self.op1(x)
        split = torch.split(out, self.chuck_size, dim=1)
        split_tensor = torch.stack(split, dim=1)
        out = torch.sum(split_tensor, dim=1)
        out = self.op2(out)
        return out


class DepthToSpace(nn.Module):

    def __init__(self, C_in, kernel_size, stride, padding):
        super(DepthToSpace, self).__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, bias=False), nn.PixelShuffle(stride), nn.Conv2d(int(C_in / stride ** 2), C_in, kernel_size=kernel_size, padding=padding, bias=False), nn.LeakyReLU(0.2, inplace=False))

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride, op_type='downsample'):
        super(Zero, self).__init__()
        self.stride = stride
        self.op_type = op_type
        if self.op_type == 'upsample':
            self.op = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        if self.op_type == 'downsample':
            return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            return self.op(x).mul(0.0)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True, op_type='downsample'):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.op_type = op_type
        if self.op_type == 'downsample':
            self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        else:
            self.op_1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.op_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        if self.op_type == 'downsample':
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        else:
            out = self.op_1(x)
        out = self.bn(out)
        return out


ACTIVATION_OPS = {'none': None, 'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(0.2, inplace=False)}


class BilinearOp(nn.Module):

    def __init__(self, stride, upsample_mode, act_op):
        super(BilinearOp, self).__init__()
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.Upsample(scale_factor=stride, mode=upsample_mode))
        else:
            self.op = nn.Sequential(nn.Upsample(scale_factor=stride, mode=upsample_mode), activation)

    def forward(self, x):
        return self.op(x)


class DepthToSpaceOp(nn.Module):

    def __init__(self, stride, act_op, affine=True):
        super(DepthToSpaceOp, self).__init__()
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.PixelShuffle(stride))
        else:
            self.op = nn.Sequential(nn.PixelShuffle(stride), activation)

    def forward(self, x):
        return self.op(x)


KERNEL_SIZE_OPS = {'1x1': 1, '3x3': 3, '4x4': 4, '5x5': 5, '7x7': 7}


PADDING_OPS = {'1x1': 0, '3x3': 1, '5x5': 2, '7x7': 3}


class TransConvOp(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, act_op, affine=True):
        super(TransConvOp, self).__init__()
        padding = PADDING_OPS[kernel_size]
        kernel_size = KERNEL_SIZE_OPS[kernel_size]
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride - 1))
        else:
            self.op = nn.Sequential(nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride - 1), activation)

    def forward(self, x):
        return self.op(x)


class ConvOp(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, act_op, affine=True):
        super(ConvOp, self).__init__()
        padding = PADDING_OPS[kernel_size]
        kernel_size = KERNEL_SIZE_OPS[kernel_size]
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False))
        else:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False), activation)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SplitStackSum(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, act_op, split=4, affine=True):
        super(SplitStackSum, self).__init__()
        padding = PADDING_OPS[kernel_size]
        kernel_size = KERNEL_SIZE_OPS[kernel_size]
        activation = ACTIVATION_OPS[act_op]
        self.chuck_size = int(C_in / split)
        if not activation:
            self.op = nn.Sequential(nn.Conv2d(int(C_in / split), C_out, kernel_size=kernel_size, padding=padding, bias=False))
        else:
            self.op = nn.Sequential(nn.Conv2d(int(C_in / split), C_out, kernel_size=kernel_size, padding=padding, bias=False), activation)

    def forward(self, x):
        split = torch.split(x, self.chuck_size, dim=1)
        stack = torch.stack(split, dim=1)
        out = torch.sum(stack, dim=1)
        out = self.op(out)
        return out


class SepConvOp(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, act_op, affine=True):
        super(SepConvOp, self).__init__()
        padding = PADDING_OPS[kernel_size]
        kernel_size = KERNEL_SIZE_OPS[kernel_size]
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        else:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), activation)

    def forward(self, x):
        return self.op(x)


class DepthWiseConvOp(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, act_op, affine=True):
        super(DepthWiseConvOp, self).__init__()
        padding = PADDING_OPS[kernel_size]
        kernel_size = KERNEL_SIZE_OPS[kernel_size]
        activation = ACTIVATION_OPS[act_op]
        if not activation:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, groups=C_out, bias=False))
        else:
            self.op = nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, groups=C_out, bias=False), activation)

    def forward(self, x):
        return self.op(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearAdditive,
     lambda: ([], {'C_in': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BilinearConv,
     lambda: ([], {'C_in': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvDownSample,
     lambda: ([], {'C_in': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'bias': 4, 'pad': 4, 'act_fun': _mock_layer, 'need1x1_up': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthToSpace,
     lambda: ([], {'C_in': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GenNoise,
     lambda: ([], {'dim2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutputBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'bias': 4, 'pad': 4, 'need_sigmoid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'num_input_channels': 4, 'num_output_channels': 4, 'num_blocks': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'bias': 4, 'pad': 4, 'act_fun': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stem,
     lambda: ([], {'num_input_channel': 4, 'num_output_channel': 4, 'norm_layer': _mock_layer, 'need_bias': 4, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransConv,
     lambda: ([], {'C_in': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (View,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (unetConv2,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'norm_layer': _mock_layer, 'need_bias': 4, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (unetDown,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'norm_layer': _mock_layer, 'need_bias': 4, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_YunChunChen_NAS_DIP_pytorch(_paritybench_base):
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

