import sys
_module = sys.modules[__name__]
del sys
load_model = _module
setup_fid = _module
submit = _module
optimizer = _module
util = _module
generate = _module
loss_criterions = _module
base_loss_criterions = _module
gradient_losses = _module
networks = _module
building_blocks = _module
custom_layers = _module
style_gan_net = _module
train = _module
utils = _module
src = _module
data = _module
imagefolder = _module
multiimagefolder = _module
nodata = _module
utils = _module
models = _module
base = _module
blobgan = _module
gan = _module
invertblobgan = _module
networks = _module
layoutnet = _module
layoutstylegan = _module
op = _module
conv2d_gradfix = _module
conv2d_gradfix_111andon = _module
conv2d_gradfix_pre111 = _module
fused_act = _module
upfirdn2d = _module
stylegan = _module
run = _module
colab = _module
distributed = _module
io = _module
logging = _module
misc = _module
training = _module
wandb_logger = _module

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


import numpy as np


from torchvision.transforms import functional as F


import collections


from matplotlib import pyplot as plt


import torchvision


import torch.nn.functional as F


import torch.nn as nn


from collections import OrderedDict


import math


from numpy import prod


from torch.nn import functional as F


import time


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


from typing import Any


from typing import Optional


from typing import Union


from typing import Dict


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.transforms import InterpolationMode


from typing import List


from typing import Callable


from torch.utils.data import Dataset


import itertools


from torch.utils.data import IterableDataset


from typing import Tuple


from torchvision.datasets.folder import default_loader


from torchvision.datasets.folder import ImageFolder


from torchvision.datasets.folder import make_dataset


from itertools import groupby


from numbers import Number


from torch import Tensor


import random


from matplotlib import cm


from torch import nn


from torch.cuda.amp import autocast


from torch.optim import Optimizer


from torchvision.utils import make_grid


import warnings


from torch import autograd


from torch.autograd import Function


from torch.utils.cpp_extension import load


from collections import abc


from torchvision.transforms.functional import gaussian_blur


from collections import defaultdict


import matplotlib.pyplot as plt


import torchvision.transforms.functional as FF


from torchvision.datasets.utils import download_url


from torch import distributed as dist


import re


import torchvision.transforms.functional as F


from math import pi


from typing import TypeVar


from typing import OrderedDict


from torchvision.transforms.functional import to_tensor


from math import sqrt


from math import ceil


import torchvision.utils as utils


class NoiseMixin(nn.Module):
    """
    Add noise with channel wise scaling factor
    reference: apply_noise in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """

    def __init__(self, num_channels):
        super(NoiseMixin, self).__init__()
        self.weight = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, noise=None):
        assert len(x.size()) == 4
        s = x.size()
        if noise is None:
            noise = torch.randn(s[0], 1, s[2], s[3], device=x.device, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class NormalizationLayer(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-08):
        return x * ((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt()


def getLayerNormalizationFactor(x, gain):
    """
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])
    return gain * math.sqrt(1.0 / fan_in)


class ConstrainedLayer(nn.Module):
    """
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self, module, use_wscale=True, lrmul=1.0, bias=True, gain=np.sqrt(2)):
        """
        use_wscale (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        init_bias_to_zero (bool): if true, bias will be initialized to zero
        """
        super(ConstrainedLayer, self).__init__()
        self.module = module
        self.equalized = use_wscale
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.module.weight.size(0)))
            self.bias_mul = 1.0
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrmul
            self.weight_mul = getLayerNormalizationFactor(self.module, gain=gain) * lrmul
            self.bias_mul = lrmul

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight_mul
        if self.bias is not None:
            if x.dim() == 2:
                x = x + self.bias.view(1, -1) * self.bias_mul
            else:
                x = x + self.bias.view(1, -1, 1, 1) * self.bias_mul
        return x


class EqualizedLinear(ConstrainedLayer):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        A nn.Linear module with specific constraints
        Args:
            in_channels (int): number of channels in the previous layer
            out_channels (int): number of channels of the current layer
            bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self, nn.Linear(in_channels, out_channels, bias=False), **kwargs)


class StyleMixin(nn.Module):
    """
    Style modulation.
    reference: style_mod in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """

    def __init__(self, dlatent_size, num_channels, use_wscale):
        super(StyleMixin, self).__init__()
        self.linear = EqualizedLinear(dlatent_size, num_channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, w):
        style = self.linear(w)
        shape = [-1, 2, x.size(1)] + [1] * (x.dim() - 2)
        style = style.view(shape)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class LayerEpilogue(nn.Module):
    """
    Things to do at the end of each layer
    1. mixin scaled noise
    2. mixin style with AdaIN
    """

    def __init__(self, num_channels, dlatent_size, use_wscale, use_pixel_norm, use_instance_norm, use_noise, use_styles, nonlinearity):
        super(LayerEpilogue, self).__init__()
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        layers = []
        if use_noise:
            layers.append(('noise', NoiseMixin(num_channels)))
        layers.append(('act', act))
        if use_pixel_norm:
            layers.append(('pixel_norm', NormalizationLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(num_channels)))
        self.pre_style_op = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMixin(dlatent_size, num_channels, use_wscale=use_wscale)

    def forward(self, x, dlatent):
        x = self.pre_style_op(x)
        if self.style_mod:
            x = self.style_mod(x, dlatent)
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, **kwargs):
        """
        A nn.Conv2d module with specific constraints
        Args:
            in_channels (int): number of channels in the previous layer
            out_channels (int): number of channels of the current layer
            kernel_size (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """
        ConstrainedLayer.__init__(self, nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False), **kwargs)


class EarlySynthesisBlock(nn.Module):
    """
    The first block for 4x4 resolution
    """

    def __init__(self, in_channels, dlatent_size, const_input_layer, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, nonlinearity):
        super(EarlySynthesisBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.in_channels = in_channels
        if const_input_layer:
            self.const = nn.Parameter(torch.ones(1, in_channels, 4, 4))
            self.bias = nn.Parameter(torch.ones(in_channels))
        else:
            self.dense = EqualizedLinear(dlatent_size, in_channels * 16, use_wscale=use_wscale)
        self.epi0 = LayerEpilogue(num_channels=in_channels, dlatent_size=dlatent_size, use_wscale=use_wscale, use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm, use_styles=use_styles, nonlinearity=nonlinearity)
        self.conv = EqualizedConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=3 // 2)
        self.epi1 = LayerEpilogue(num_channels=in_channels, dlatent_size=dlatent_size, use_wscale=use_wscale, use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm, use_styles=use_styles, nonlinearity=nonlinearity)

    def forward(self, dlatents):
        dlatents_0 = dlatents[:, 0]
        dlatents_1 = dlatents[:, 1]
        batch_size = dlatents.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_0).view(batch_size, self.in_channels, 4, 4)
        x = self.epi0(x, dlatents_0)
        x = self.conv(x)
        x = self.epi1(x, dlatents_1)
        return x


class Blur2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """

    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2d, self).__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1))
        return x


class SmoothUpsample(nn.Module):
    """
    https://arxiv.org/pdf/1904.11486.pdf
    'Making Convolutional Networks Shift-Invariant Again'
    # this is in the tf implementation too
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SmoothUpsample, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        weight = self.weight.permute([1, 0, 2, 3])
        weight = F.pad(weight, [1, 1, 1, 1])
        weight = weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]
        x = F.conv_transpose2d(x, weight, self.bias, stride=2, padding=self.padding)
        return x


class EqualizedSmoothUpsample(ConstrainedLayer):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        ConstrainedLayer.__init__(self, SmoothUpsample(in_channels, out_channels, kernel_size=kernel_size, bias=False), **kwargs)


def _upscale2d(x, factor):
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


class Upscale2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """

    def __init__(self, factor=2):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        return _upscale2d(x, self.factor)


class Upscale2dConv2d(nn.Module):

    def __init__(self, res, in_channels, out_channels, kernel_size, use_wscale, fused_scale='auto', **kwargs):
        super(Upscale2dConv2d, self).__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.fused_scale = fused_scale
        self.upscale = Upscale2d()
        if self.fused_scale == 'auto':
            self.fused_scale = 2 ** res >= 128
        if not self.fused_scale:
            self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, use_wscale=use_wscale)
        else:
            self.conv = EqualizedSmoothUpsample(in_channels, out_channels, kernel_size, use_wscale=use_wscale)

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.upscale(x))
        else:
            return self.conv(x)


class LaterSynthesisBlock(nn.Module):
    """
    The following blocks for res 8x8...etc.
    """

    def __init__(self, in_channels, out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, nonlinearity, blur_filter, res):
        super(LaterSynthesisBlock, self).__init__()
        assert isinstance(res, int) and 2 <= res <= 10
        self.res = res
        if blur_filter:
            self.blur = Blur2d(blur_filter)
        else:
            self.blur = None
        self.conv0_up = Upscale2dConv2d(res=res, in_channels=in_channels, out_channels=out_channels, kernel_size=3, use_wscale=use_wscale)
        self.epi0 = LayerEpilogue(num_channels=out_channels, dlatent_size=dlatent_size, use_wscale=use_wscale, use_pixel_norm=use_pixel_norm, use_noise=use_noise, use_instance_norm=use_instance_norm, use_styles=use_styles, nonlinearity=nonlinearity)
        self.conv1 = EqualizedConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=3 // 2)
        self.epi1 = LayerEpilogue(num_channels=out_channels, dlatent_size=dlatent_size, use_wscale=use_wscale, use_pixel_norm=use_pixel_norm, use_noise=use_noise, use_instance_norm=use_instance_norm, use_styles=use_styles, nonlinearity=nonlinearity)

    def forward(self, x, dlatents):
        x = self.conv0_up(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.epi0(x, dlatents[:, self.res * 2 - 4])
        x = self.conv1(x)
        x = self.epi1(x, dlatents[:, self.res * 2 - 3])
        return x


class Downscale2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """

    def __init__(self, factor=2):
        super(Downscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        factor = self.factor
        if factor == 1:
            return x
        return F.avg_pool2d(x, factor)


class SmoothDownsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SmoothDownsample, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        weight = F.pad(self.weight, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4
        x = F.conv2d(x, weight, stride=2, padding=self.padding)
        return x


class EqualizedSmoothDownsample(ConstrainedLayer):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        ConstrainedLayer.__init__(self, SmoothDownsample(in_channels, out_channels, kernel_size=kernel_size, bias=False), **kwargs)


class Downscale2dConv2d(nn.Module):

    def __init__(self, res, in_channels, out_channels, kernel_size, use_wscale, fused_scale, **kwargs):
        super(Downscale2dConv2d, self).__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.fused_scale = fused_scale
        self.downscale = Downscale2d()
        if self.fused_scale == 'auto':
            self.fused_scale = 2 ** res >= 128
        if not self.fused_scale:
            self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, use_wscale=use_wscale)
        else:
            self.conv = EqualizedSmoothDownsample(in_channels, out_channels, kernel_size, use_wscale=use_wscale)

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.downscale(x))
        else:
            return self.conv(x)


class EarlyDiscriminatorBlock(nn.Sequential):

    def __init__(self, res, in_channels, out_channels, use_wscale, blur_filter, fused_scale, nonlinearity):
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        layers = []
        layers.append(('conv0', EqualizedConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=3 // 2, use_wscale=use_wscale)))
        layers.append(('act0', act))
        layers.append(('blur', Blur2d(blur_filter)))
        layers.append(('conv1_down', Downscale2dConv2d(res=res, in_channels=in_channels, out_channels=out_channels, kernel_size=3, fused_scale=fused_scale, use_wscale=use_wscale)))
        layers.append(('act1', act))
        super().__init__(OrderedDict(layers))


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class MiniBatchStdDev(nn.Module):
    """
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """

    def __init__(self, subgroup_size, num_features):
        super(MiniBatchStdDev, self).__init__()
        self.subgroup_size = subgroup_size
        self.num_features = num_features

    def forward(self, x):
        s = x.size()
        subgroup_size = min(s[0], self.subgroup_size)
        if s[0] % subgroup_size != 0:
            subgroup_size = s[0]
        if subgroup_size > 1:
            y = x.view(subgroup_size, -1, self.num_features, s[1] // self.num_features, s[2], s[3])
            y = y - y.mean(0, keepdim=True)
            y = (y ** 2).mean(0, keepdim=True)
            y = (y + 1e-08) ** 0.5
            y = y.mean([3, 4, 5], keepdim=True).squeeze(3)
            y = y.expand(subgroup_size, -1, -1, s[2], s[3]).contiguous().reshape(s[0], self.num_features, s[2], s[3])
        else:
            y = torch.zeros(x.size(0), self.num_features, x.size(2), x.size(3), device=x.device)
        return torch.cat([x, y], dim=1)


class LaterDiscriminatorBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_wscale, nonlinearity, mbstd_group_size, mbstd_num_features, res):
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        resolution = 2 ** res
        layers = []
        layers.append(('minibatchstddev', MiniBatchStdDev(mbstd_group_size, mbstd_num_features)))
        layers.append(('conv', EqualizedConv2d(in_channels=in_channels + mbstd_num_features, out_channels=in_channels, kernel_size=3, padding=3 // 2, use_wscale=use_wscale)))
        layers.append(('act0', act))
        layers.append(('flatten', Flatten()))
        layers.append(('dense0', EqualizedLinear(in_channels=in_channels * resolution ** 2, out_channels=in_channels, use_wscale=use_wscale)))
        layers.append(('act1', act))
        layers.append(('dense1', EqualizedLinear(in_channels=in_channels, out_channels=out_channels)))
        super().__init__(OrderedDict(layers))


class MappingNet(nn.Sequential):
    """
    A mapping network f implemented using an 8-layer MLP
    """

    def __init__(self, resolution=1024, num_layers=8, dlatent_size=512, normalize_latents=True, nonlinearity='lrelu', maping_lrmul=0.01, **kwargs):
        resolution_log2: int = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and 4 <= resolution <= 1024
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        self.dlatent_broadcast = resolution_log2 * 2 - 2
        layers = []
        if normalize_latents:
            layers.append(('pixel_norm', NormalizationLayer()))
        for i in range(num_layers):
            layers.append(('dense{}'.format(i), EqualizedLinear(dlatent_size, dlatent_size, use_wscale=True, lrmul=maping_lrmul)))
            layers.append(('dense{}_act'.format(i), act))
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        w = super().forward(x)
        if self.dlatent_broadcast is not None:
            w = w.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return w


class SynthesisNet(nn.Module):
    """
    Synthesis network
    """

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu', use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=[1, 2, 1], **kwargs):
        super(SynthesisNet, self).__init__()
        resolution_log2: int = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and 4 <= resolution <= 1024

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        blocks = []
        torgbs = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            block_name = '{s}x{s}'.format(s=2 ** res)
            torgb_name = 'torgb_lod{}'.format(resolution_log2 - res)
            if res == 2:
                block = block_name, EarlySynthesisBlock(channels, dlatent_size, const_input_layer, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, nonlinearity)
            else:
                block = block_name, LaterSynthesisBlock(last_channels, out_channels=channels, dlatent_size=dlatent_size, use_wscale=use_wscale, use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm, use_styles=use_styles, nonlinearity=nonlinearity, blur_filter=blur_filter, res=res)
            torgb = torgb_name, EqualizedConv2d(channels, num_channels, 1, use_wscale=use_wscale)
            blocks.append(block)
            torgbs.append(torgb)
            last_channels = channels
        self.torgbs = nn.ModuleDict(OrderedDict(torgbs))
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents, res, alpha):
        assert 2 <= res <= 10
        step = res - 1
        block_list = list(self.blocks.values())[:step]
        torgb_list = list(self.torgbs.values())[:step]
        if step > 1:
            skip_torgb = torgb_list[-2]
        this_rgb = torgb_list[-1]
        for i, block in enumerate(block_list):
            if i == 0:
                x = block(dlatents)
            else:
                x = block(x, dlatents)
            if i == step - 2:
                skip_x = _upscale2d(skip_torgb(x), 2)
        x = this_rgb(x)
        x = (1 - alpha) * skip_x + alpha * x
        return x


class Generator(nn.Sequential):

    def __init__(self, **kwargs):
        super().__init__(OrderedDict([('g_mapping', MappingNet(**kwargs)), ('g_synthesis', SynthesisNet(**kwargs))]))

    def forward(self, latents, res, alpha):
        dlatents = self.g_mapping(latents)
        x = self.g_synthesis(dlatents, res, alpha)
        return x


class BasicDiscriminator(nn.Module):

    def __init__(self, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512, nonlinearity='lrelu', mbstd_group_size=4, mbstd_num_features=1, use_wscale=True, fused_scale='auto', blur_filter=[1, 2, 1]):
        super(BasicDiscriminator, self).__init__()
        resolution_log2: int = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and 4 <= resolution <= 1024

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        act = {'relu': torch.relu, 'lrelu': nn.LeakyReLU(negative_slope=0.2)}[nonlinearity]
        blocks = []
        fromrgbs = []
        for res in range(resolution_log2, 1, -1):
            block_name = '{s}x{s}'.format(s=2 ** res)
            fromrgb_name = 'fromrgb_lod{}'.format(resolution_log2 - res)
            if res != 2:
                blocks.append((block_name, EarlyDiscriminatorBlock(res=res, in_channels=nf(res - 1), out_channels=nf(res - 2), use_wscale=use_wscale, blur_filter=blur_filter, fused_scale=fused_scale, nonlinearity=nonlinearity)))
            else:
                blocks.append((block_name, LaterDiscriminatorBlock(in_channels=nf(res), out_channels=1, mbstd_group_size=mbstd_group_size, mbstd_num_features=mbstd_num_features, use_wscale=use_wscale, nonlinearity=nonlinearity, res=2)))
            fromrgbs.append((fromrgb_name, EqualizedConv2d(num_channels, nf(res - 1), 1, use_wscale=use_wscale)))
        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        self.fromrgbs = nn.ModuleDict(OrderedDict(fromrgbs))

    def forward(self, x, res, alpha):
        assert 2 <= res <= 10
        step = res - 1
        block_list = list(self.blocks.values())[-step:]
        fromrgb_list = list(self.fromrgbs.values())[-step:]
        if step > 1:
            skip_fromrgb = fromrgb_list[1]
        this_fromrgb = fromrgb_list[0]
        for i, block in enumerate(block_list):
            if i == 0:
                skip_x = skip_fromrgb(F.avg_pool2d(x, 2))
                x = block(this_fromrgb(x))
                x = (1 - alpha) * skip_x + alpha * x
            else:
                x = block(x)
        return x


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)
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
        gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        ctx.bias = bias is not None
        if bias is None:
            bias = empty
        out = fused.fused_bias_act(input.float(), bias, empty.float(), 3, 0, negative_slope, scale)
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
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


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
        assert input.shape[-1] == self.weight.shape[-1], f'Input shape {input.shape[-1]} != weight shape {self.weight.shape[-1]}'
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {'' if self.bias is not None else 'no '}bias, lr_mul={self.lr_mul}, act={self.activation})"


def pixel_norm(x):
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-05)


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return pixel_norm(x)


def StyleMLP(n_layers, dim, lr_mul=1, first_dim=None, last_dim=None, pixel_norm=True, last_relu=True):
    if first_dim is None:
        first_dim = dim
    if last_dim is None:
        last_dim = dim
    _layers: list[nn.Module] = [PixelNorm()] if pixel_norm else []
    for i in range(n_layers):
        _layers.append(EqualLinear(dim if i else first_dim, dim if i + 1 < n_layers else last_dim, activation='fused_lrelu' if i + 1 < n_layers or last_relu else False, lr_mul=lr_mul))
    return nn.Sequential(*_layers)


def get_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def is_rank_zero():
    return get_rank() == 0


def derangement(n: int) ->Tensor:
    global DERANGEMENT_WARNED
    orig = torch.arange(n)
    shuffle = torch.randperm(n)
    if n == 1 and not DERANGEMENT_WARNED:
        if is_rank_zero():
            None
        DERANGEMENT_WARNED = True
    while n > 1 and (shuffle == orig).any():
        shuffle = torch.randperm(n)
    return shuffle


def derange_tensor(x: Tensor, dim: int=0) ->Tensor:
    if dim == 0:
        return x[derangement(len(x))]
    elif dim == 1:
        return x[:, derangement(len(x[0]))]


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
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


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


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
        grad_input = upfirdn2d_op.upfirdn2d(grad_output.float(), grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
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
        gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input.float(), kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
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
        out = upfirdn2d_op.upfirdn2d(input.float(), kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
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
                out = F.conv2d(input=input, weight=weight, bias=bias if bias is not None else bias, **common_kwargs)
            else:
                out = F.conv_transpose2d(input=input, weight=weight, bias=bias if bias else bias, output_padding=output_padding, **common_kwargs)
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


TORCH_EINSUM = True


einsum = torch.einsum if TORCH_EINSUM else oe.contract


def splat_features_from_scores(scores: Tensor, features: Tensor, size: Optional[int], channels_last: bool=True) ->Tensor:
    """

    Args:
        channels_last: expect input with M at end or not, see below
        scores: [N, H, W, M] (or [N, M, H, W] if not channels last)
        features: [N, M, C]
        size: dimension of map to return
    Returns: [N, C, H, W]

    """
    if size and not scores.shape[2] == size:
        if channels_last:
            scores = einops.rearrange(scores, 'n h w m -> n m h w')
        scores = F.interpolate(scores, size, mode='bilinear', align_corners=False)
        einstr = 'nmhw,nmc->nchw'
    else:
        einstr = 'nhwm,nmc->nchw' if channels_last else 'nmhw,nmc->nchw'
    return einsum(einstr, scores, features).contiguous()


class SpatialModulatedConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], fused=True):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.style_dim = style_dim
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
        spatial_style = isinstance(style, dict)
        if spatial_style or not self.fused:
            if spatial_style:
                layout = style
                style = layout['spatial_style']
                style = self.modulation(style.flatten(end_dim=1)).view(*style.shape[:-1], -1)
                style = splat_features_from_scores(layout['scores_pyramid'][input.size(-1)], style, input.size(-1), channels_last=False)
                if self.demodulate:
                    style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-08)
            else:
                style = self.modulation(style).reshape(batch, in_channel, 1, 1)
            weight = self.scale * self.weight.squeeze(0)
            input = input * style
            if self.demodulate:
                if spatial_style:
                    demod = torch.rsqrt(weight.unsqueeze(0).pow(2).sum([2, 3, 4]) + 1e-08)
                    weight = weight * demod.view(self.out_channel, 1, 1, 1)
                else:
                    w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                    dcoefs = (w.square().sum((2, 3, 4)) + 1e-08).rsqrt()
            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(input, weight, padding=0, stride=2)
                out = self.blur(out)
            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)
            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)
            if self.demodulate and not style.dim() > 2:
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


class SpatialStyledConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv = SpatialModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
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

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class SpatialToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], c_out=3):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = SpatialModulatedConv2d(in_channel, c_out, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
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


class EqualConv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size).div_(lr_mul))
        self.scale = lr_mul / math.sqrt(in_channel * kernel_size)
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input: Tensor):
        out = F.conv1d(input, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


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


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], c_out=3):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, c_out, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class StyleGANGenerator(nn.Module):

    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, c_out=3, latent_to_img_space='w', **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.size = size
        self.latent_to_img_space = latent_to_img_space
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)
        self.channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, c_out=c_out)
        self.log_size = int(math.log(size, 2))
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
            self.to_rgbs.append(ToRGB(out_channel, style_dim, c_out=c_out))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2

    def latent_to_img(self, latent: Tensor) ->Tensor:
        if self.latent_to_img_space == 'z':
            return self([latent], return_image_only=True)
        elif self.latent_to_img_space in ['w']:
            return self([latent], input_is_latent=True, return_image_only=True)
        else:
            raise ValueError('w+ not supported yet, need to reshape, etc.')

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

    def forward(self, styles, return_latents=False, return_image_only=False, inject_index=None, truncation=1, truncation_latent=None, input_is_latent=False, noise=None, randomize_noise=True, return_features=False):
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
        image = skip
        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        elif return_image_only:
            return image
        else:
            return image, None


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


class StyleGANDiscriminator(nn.Module):

    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], discriminate_stddev=True, in_channels=3, blur_input=False, blur_kernel_size=3, blur_sigma=1, d_out=1):
        super().__init__()
        self.discriminate_stddev = discriminate_stddev
        self.blur_input = blur_input
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        channels = {(4): 512 * channel_multiplier // 2, (8): 512 * channel_multiplier // 2, (16): 512 * channel_multiplier // 2, (32): 512 * channel_multiplier // 2, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        convs = [ConvLayer(in_channels, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + (1 if discriminate_stddev else 0), channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], d_out))

    def forward(self, input):
        if self.blur_input:
            input = gaussian_blur(input, self.blur_kernel_size, self.blur_sigma)
        out = self.convs(input)
        batch, channel, height, width = out.shape
        if self.discriminate_stddev:
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
    (EqualizedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualizedLinear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MiniBatchStdDev,
     lambda: ([], {'subgroup_size': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoiseMixin,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormalizationLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyleMixin,
     lambda: ([], {'dlatent_size': 4, 'num_channels': 4, 'use_wscale': 1.0}),
     lambda: ([torch.rand([128, 4, 4, 4]), torch.rand([4, 8, 4, 4])], {}),
     True),
    (Upscale2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upscale2dConv2d,
     lambda: ([], {'res': 4, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 1, 'use_wscale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_dave_epstein_blobgan(_paritybench_base):
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

