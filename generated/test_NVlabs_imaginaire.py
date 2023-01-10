import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
imaginaire = _module
config = _module
datasets = _module
base = _module
cache = _module
dummy = _module
folder = _module
images = _module
lmdb = _module
object_store = _module
paired_few_shot_videos = _module
paired_few_shot_videos_native = _module
paired_images = _module
paired_videos = _module
unpaired_few_shot_images = _module
unpaired_images = _module
discriminators = _module
dummy = _module
fpse = _module
fs_vid2vid = _module
funit = _module
gancraft = _module
mlp_multiclass = _module
multires_patch = _module
munit = _module
residual = _module
spade = _module
unit = _module
evaluation = _module
caption = _module
clip = _module
common = _module
r_precision = _module
common = _module
fid = _module
kid = _module
knn = _module
lpips = _module
msid = _module
prdc = _module
pretrained = _module
segmentation = _module
celebamask_hq = _module
cocostuff = _module
common = _module
generators = _module
coco_funit = _module
dummy = _module
fs_vid2vid = _module
funit = _module
gancraft = _module
gancraft_base = _module
munit = _module
pix2pixHD = _module
spade = _module
unit = _module
vid2vid = _module
wc_vid2vid = _module
layers = _module
activation_norm = _module
conv = _module
misc = _module
non_local = _module
nonlinearity = _module
residual = _module
residual_deep = _module
vit = _module
weight_norm = _module
losses = _module
dict = _module
feature_matching = _module
flow = _module
gan = _module
info_nce = _module
kl = _module
perceptual = _module
weighted_mse = _module
model_utils = _module
fs_vid2vid = _module
camctl = _module
layers = _module
loss = _module
mc_lbl_reduction = _module
mc_utils = _module
voxlib = _module
positional_encoding = _module
setup = _module
sp_trilinear = _module
label = _module
pix2pixHD = _module
rename_inputs = _module
render = _module
optimizers = _module
fromage = _module
madam = _module
third_party = _module
bias_act = _module
bias_act = _module
setup = _module
channelnorm = _module
setup = _module
correlation = _module
setup = _module
flow_net = _module
flow_net = _module
models = _module
networks = _module
flownet_c = _module
flownet_fusion = _module
flownet_s = _module
flownet_sd = _module
submodules = _module
utils = _module
flow_utils = _module
frame_utils = _module
param_utils = _module
tools = _module
test_flownet2 = _module
resample2d = _module
setup = _module
upfirdn2d = _module
setup = _module
upfirdn2d = _module
trainers = _module
base = _module
fs_vid2vid = _module
funit = _module
gancraft = _module
munit = _module
pix2pixHD = _module
spade = _module
unit = _module
vid2vid = _module
wc_vid2vid = _module
cudnn = _module
data = _module
dataset = _module
diff_aug = _module
distributed = _module
gpu_affinity = _module
init_weight = _module
io = _module
logging = _module
meters = _module
misc = _module
model_average = _module
path = _module
trainer = _module
visualization = _module
common = _module
face = _module
pose = _module
inference = _module
sch2vox = _module
build_index = _module
build_lmdb = _module
download_dataset = _module
download_test_data = _module
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


from collections import OrderedDict


from functools import partial


from inspect import signature


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import copy


import random


import warnings


import torch.nn as nn


import functools


import torch.nn.functional as F


from torch import nn


from time import sleep


from typing import Union


from typing import List


from typing import Tuple


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torch import distributed as dist


from torch.nn import functional as F


from torch.distributed import barrier


import math


import torch.distributed as dist


from torchvision.models import inception_v3


from scipy import linalg


from collections import namedtuple


import torchvision.models as tv


from scipy.sparse import lil_matrix


from scipy.sparse import diags


from scipy.sparse import eye


from torchvision.models import inception


from torchvision.models import vgg16


import torch.hub


from types import SimpleNamespace


import re


from torch.nn import Upsample as NearestUpsample


import types


from torchvision import transforms


from torch.utils.checkpoint import checkpoint


import collections


from torch.nn.utils import spectral_norm


from torch.nn.utils import weight_norm


from torch.nn.utils.spectral_norm import SpectralNorm


from torch.nn.utils.spectral_norm import SpectralNormStateDictHook


from torch.nn.utils.spectral_norm import SpectralNormLoadStateDictPreHook


import torchvision


import matplotlib.pyplot as plt


import time


from scipy import ndimage


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from sklearn.cluster import KMeans


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.autograd import Variable


from torch.nn.modules.module import Module


from torch.nn import init


from inspect import isclass


import inspect


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import torch.backends.cudnn as cudnn


import torchvision.utils


from torch.utils.tensorboard import SummaryWriter


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim import lr_scheduler


from scipy.optimize import curve_fit


from scipy.signal import medfilt


import torch.autograd.profiler as profiler


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    """Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * gain ** (f.ndim / 2)
    f = f
    return f


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    assert fw >= 1 and fh >= 1
    return fw, fh


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    key = upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]


    class Upfirdn2dCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, f):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, np.sqrt(gain))
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [fw - padx0 - 1, iw * upx - ow * downx + padx0 - upx + 1, fh - pady0 - 1, ih * upy - oh * downy + pady0 - upy + 1]
            dx = None
            df = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=not flip_filter, gain=gain).apply(dy, f)
            assert not ctx.needs_input_grad[1]
            return dx, df
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda


def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0):x.shape[2] - max(-pady1, 0), max(-padx0, 0):x.shape[3] - max(-padx1, 0)]
    f = f * gain ** (f.ndim / 2)
    f = f
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = F.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = F.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = F.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    """Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda':
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


class BlurDownsample(nn.Module):

    def __init__(self, kernel=(1, 3, 3, 1), factor=2, padding_mode='zeros'):
        super().__init__()
        p = len(kernel)
        px0 = (p - factor + 1) // 2
        px1 = (p - factor) // 2
        py0 = (p - factor + 1) // 2
        py1 = (p - factor) // 2
        self.pad = [px0, px1, py0, py1]
        self.factor = factor
        self.register_buffer('kernel', setup_filter(kernel))
        self.kernel_1d = kernel
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, list(self.pad) * 2, mode=self.padding_mode)
            out = upfirdn2d(x, self.kernel, down=self.factor)
        else:
            out = upfirdn2d(x, self.kernel, down=self.factor, padding=self.pad)
        return out

    def extra_repr(self):
        s = 'kernel={kernel_1d}, padding_mode={padding_mode}, pad={pad}'
        return s.format(**self.__dict__)


class ApplyNoise(nn.Module):
    """Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.conditional = True

    def forward(self, x, *_args, noise=None, **_kwargs):
        """

        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()
        return x + self.scale * noise


class Blur(nn.Module):

    def __init__(self, kernel=(1, 3, 3, 1), pad=0, padding_mode='zeros'):
        super().__init__()
        self.register_buffer('kernel', setup_filter(kernel))
        self.kernel_1d = kernel
        self.padding_mode = padding_mode
        self.pad = pad

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, list(self.pad) * 2, mode=self.padding_mode)
            out = upfirdn2d(x, self.kernel)
        else:
            out = upfirdn2d(x, self.kernel, padding=self.pad)
        return out

    def extra_repr(self):
        s = 'kernel={kernel_1d}, padding_mode={padding_mode}, pad={pad}'
        return s.format(**self.__dict__)


class _BaseConvBlock(nn.Module):
    """An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, input_dim, clamp, blur_kernel, output_scale, init_gain):
        super().__init__()
        self.weight_norm_type = weight_norm_type
        self.stride = stride
        self.clamp = clamp
        self.init_gain = init_gain
        if 'fused' in nonlinearity:
            lr_mul = getattr(weight_norm_params, 'lr_mul', 1)
            conv_before_nonlinearity = order.find('C') < order.find('A')
            if conv_before_nonlinearity:
                assert bias is True
                bias = False
            channel = out_channels if conv_before_nonlinearity else in_channels
            nonlinearity_layer = get_nonlinearity_layer(nonlinearity, inplace=inplace_nonlinearity, num_channels=channel, lr_mul=lr_mul)
        else:
            nonlinearity_layer = get_nonlinearity_layer(nonlinearity, inplace=inplace_nonlinearity)
        if apply_noise:
            order = order.replace('C', 'CG')
            noise_layer = ApplyNoise()
        else:
            noise_layer = None
        if blur:
            assert blur_kernel is not None
            if stride == 2:
                p = len(blur_kernel) - 2 + (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2, p // 2
                padding = 0
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode)
                order = order.replace('C', 'BC')
            elif stride == 0.5:
                padding = 0
                p = len(blur_kernel) - 2 - (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2 + 1, p // 2 + 1
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode)
                order = order.replace('C', 'CB')
            elif stride == 1:
                blur_layer = nn.Identity()
            else:
                raise NotImplementedError
        else:
            blur_layer = nn.Identity()
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim))
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(norm_channels, activation_norm_type, input_dim, **vars(activation_norm_params))
        mappings = {'C': {'conv': conv_layer}, 'N': {'norm': activation_norm_layer}, 'A': {'nonlinearity': nonlinearity_layer}}
        mappings.update({'B': {'blur': blur_layer}})
        mappings.update({'G': {'noise': noise_layer}})
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])
        self.conditional = getattr(conv_layer, 'conditional', False) or getattr(activation_norm_layer, 'conditional', False)
        if output_scale is not None:
            self.output_scale = nn.Parameter(torch.tensor(output_scale))
        else:
            self.register_parameter('output_scale', None)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for key, layer in self.layers.items():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
            if self.clamp is not None and isinstance(layer, nn.Conv2d):
                x.clamp_(max=self.clamp)
            if key == 'conv':
                if self.output_scale is not None:
                    x = x * self.output_scale
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            if stride < 1:
                padding_mode = 'zeros'
                assert padding == 0
                layer_type = getattr(nn, f'ConvTranspose{input_dim}d')
                stride = round(1 / stride)
            else:
                layer_type = getattr(nn, f'Conv{input_dim}d')
            layer = layer_type(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        return layer

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and self.weight_norm_type != '':
                mod_str = mod_str[:-1] + ', weight_norm={}'.format(self.weight_norm_type) + ')'
            if name == 'conv' and getattr(layer, 'base_lr_mul', 1) != 1:
                mod_str = mod_str[:-1] + ', lr_mul={}'.format(layer.base_lr_mul) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ' + line) for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class Conv2dBlock(_BaseConvBlock):
    """A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, blur=False, order='CNA', clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, 2, clamp, blur_kernel, output_scale, init_gain)


class BlurUpsample(nn.Module):

    def __init__(self, kernel=(1, 3, 3, 1), factor=2, padding_mode='zeros'):
        super().__init__()
        p = len(kernel)
        px0 = (p + factor - 1) // 2
        px1 = (p - factor) // 2
        py0 = (p + factor - 1) // 2
        py1 = (p - factor) // 2
        self.pad = [px0, px1, py0, py1]
        self.factor = factor
        self.register_buffer('kernel', setup_filter(kernel))
        self.kernel_1d = kernel
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, list(self.pad) * 2, mode=self.padding_mode)
            out = upfirdn2d(x, self.kernel, up=self.factor, gain=self.factor ** 2)
        else:
            out = upfirdn2d(x, self.kernel, up=self.factor, padding=self.pad, gain=self.factor ** 2)
        return out

    def extra_repr(self):
        s = 'kernel={kernel_1d}, padding_mode={padding_mode}, pad={pad}'
        return s.format(**self.__dict__)


class _BaseResBlock(nn.Module):
    """An abstract class for residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale, skip_block=None, blur=False, upsample_first=True, skip_weight_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale = output_scale
        self.upsample_first = upsample_first
        self.stride = stride
        self.blur = blur
        if skip_block is None:
            skip_block = block
        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        if learn_shortcut is None:
            self.learn_shortcut = in_channels != out_channels
        else:
            self.learn_shortcut = learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)
        residual_params = {}
        shortcut_params = {}
        base_params = dict(dilation=dilation, groups=groups, padding_mode=padding_mode, clamp=clamp)
        residual_params.update(base_params)
        residual_params.update(dict(activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params, weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params, padding=padding, apply_noise=apply_noise))
        shortcut_params.update(base_params)
        shortcut_params.update(dict(kernel_size=1))
        if skip_activation_norm:
            shortcut_params.update(dict(activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params, apply_noise=False))
        if skip_weight_norm:
            shortcut_params.update(dict(weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params))
        if order.find('A') < order.find('C') and (activation_norm_type == '' or activation_norm_type == 'none'):
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity
        first_stride, second_stride, shortcut_stride, first_blur, second_blur, shortcut_blur = self._get_stride_blur()
        self.conv_block_0 = block(in_channels, hidden_channels, kernel_size=kernel_size, bias=biases[0], nonlinearity=nonlinearity, order=order[0:3], inplace_nonlinearity=first_inplace, stride=first_stride, blur=first_blur, **residual_params)
        self.conv_block_1 = block(hidden_channels, out_channels, kernel_size=kernel_size, bias=biases[1], nonlinearity=nonlinearity, order=order[3:], inplace_nonlinearity=inplace_nonlinearity, stride=second_stride, blur=second_blur, **residual_params)
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels, bias=biases[2], nonlinearity=skip_nonlinearity_type, order=order[0:3], stride=shortcut_stride, blur=shortcut_blur, **shortcut_params)
        elif in_channels < out_channels:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels - in_channels, bias=biases[2], nonlinearity=skip_nonlinearity_type, order=order[0:3], stride=shortcut_stride, blur=shortcut_blur, **shortcut_params)
        self.conditional = getattr(self.conv_block_0, 'conditional', False) or getattr(self.conv_block_1, 'conditional', False)

    def _get_stride_blur(self):
        if self.stride > 1:
            first_stride, second_stride = 1, self.stride
            first_blur, second_blur = False, self.blur
            shortcut_stride = self.stride
            shortcut_blur = self.blur
            self.upsample = None
        elif self.stride < 1:
            first_stride, second_stride = self.stride, 1
            first_blur, second_blur = self.blur, False
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                self.upsample = BlurUpsample()
            else:
                shortcut_stride = self.stride
                self.upsample = nn.Upsample(scale_factor=2)
        else:
            first_stride = second_stride = 1
            first_blur = second_blur = False
            shortcut_stride = 1
            shortcut_blur = False
            self.upsample = None
        return first_stride, second_stride, shortcut_stride, first_blur, second_blur, shortcut_blur

    def conv_blocks(self, x, *cond_inputs, separate_cond=False, **kw_cond_inputs):
        """Returns the output of the residual branch.

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            dx (tensor): Output tensor.
        """
        if separate_cond:
            dx = self.conv_block_0(x, cond_inputs[0], **kw_cond_inputs.get('kwargs_0', {}))
            dx = self.conv_block_1(dx, cond_inputs[1], **kw_cond_inputs.get('kwargs_1', {}))
        else:
            dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, separate_cond=False, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            do_checkpoint (bool, optional, default=``False``) If ``True``,
                trade compute for memory by checkpointing the model.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, separate_cond=separate_cond, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, separate_cond=separate_cond, **kw_cond_inputs)
        if self.upsample_first and self.upsample is not None:
            x = self.upsample(x)
        if self.learn_shortcut:
            if separate_cond:
                x_shortcut = self.conv_block_s(x, cond_inputs[2], **kw_cond_inputs.get('kwargs_2', {}))
            else:
                x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        elif self.in_channels < self.out_channels:
            if separate_cond:
                x_shortcut_pad = self.conv_block_s(x, cond_inputs[2], **kw_cond_inputs.get('kwargs_2', {}))
            else:
                x_shortcut_pad = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
            x_shortcut = torch.cat((x, x_shortcut_pad), dim=1)
        elif self.in_channels > self.out_channels:
            x_shortcut = x[:, :self.out_channels, :, :]
        else:
            x_shortcut = x
        if not self.upsample_first and self.upsample is not None:
            x_shortcut = self.upsample(x_shortcut)
        output = x_shortcut + dx
        return self.output_scale * output

    def extra_repr(self):
        s = 'output_scale={output_scale}'
        return s.format(**self.__dict__)


class Res2dBlock(_BaseResBlock):
    """Residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, skip_weight_norm=True, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1, blur=False, upsample_first=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv2dBlock, learn_shortcut, clamp, output_scale, blur=blur, upsample_first=upsample_first, skip_weight_norm=skip_weight_norm)


class ResDiscriminator(nn.Module):
    """Global residual discriminator.

    Args:
        image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        max_num_filters (int): Maximum num. of filters in a layer.
        first_kernel_size (int): Kernel size in the first layer.
        num_layers (int): Num. of layers in discriminator.
        padding_mode (str): Padding mode.
        activation_norm_type (str): Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        aggregation (str): Method to aggregate features across different
            locations in the final layer. ``'conv'``, or ``'pool'``.
        order (str): Order of operations in the residual link.
        anti_aliased (bool): If ``True``, uses anti-aliased pooling.
    """

    def __init__(self, image_channels=3, num_filters=64, max_num_filters=512, first_kernel_size=1, num_layers=4, padding_mode='zeros', activation_norm_type='', weight_norm_type='', aggregation='conv', order='pre_act', anti_aliased=False, **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn('Discriminator argument {} is not used'.format(key))
        conv_params = dict(padding_mode=padding_mode, activation_norm_type=activation_norm_type, weight_norm_type=weight_norm_type, nonlinearity='leakyrelu')
        first_padding = (first_kernel_size - 1) // 2
        model = [Conv2dBlock(image_channels, num_filters, first_kernel_size, 1, first_padding, **conv_params)]
        for _ in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model.append(Res2dBlock(num_filters_prev, num_filters, order=order, **conv_params))
            if anti_aliased:
                model.append(BlurDownsample())
            else:
                model.append(nn.AvgPool2d(2, stride=2))
        if aggregation == 'pool':
            model += [torch.nn.AdaptiveAvgPool2d(1)]
        elif aggregation == 'conv':
            model += [Conv2dBlock(num_filters, num_filters, 4, 1, 0, nonlinearity='leakyrelu')]
        else:
            raise ValueError('The aggregation mode is not recognized' % self.aggregation)
        self.model = nn.Sequential(*model)
        self.classifier = nn.Linear(num_filters, 1)

    def forward(self, images):
        """Multi-resolution patch discriminator forward.

        Args:
            images (tensor) : Input images.
        Returns:
            (tuple):
              - outputs (tensor): Output of the discriminator.
              - features (tensor): Intermediate features of the discriminator.
              - images (tensor): Input images.
        """
        batch_size = images.size(0)
        features = self.model(images)
        outputs = self.classifier(features.view(batch_size, -1))
        return outputs, features, images


class NLayerPatchDiscriminator(nn.Module):
    """Patch Discriminator constructor.

    Args:
        kernel_size (int): Convolution kernel size.
        num_input_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self, kernel_size, num_input_channels, num_filters, num_layers, max_num_filters, activation_norm_type, weight_norm_type):
        super(NLayerPatchDiscriminator, self).__init__()
        self.num_layers = num_layers
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity=nonlinearity, order='CNA')
        layers = [[base_conv2d_block(num_input_channels, num_filters, stride=2)]]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < num_layers - 1 else 1
            layers += [[base_conv2d_block(num_filters_prev, num_filters, stride=stride)]]
        layers += [[Conv2dBlock(num_filters, 1, 3, 1, padding, weight_norm_type=weight_norm_type)]]
        for n in range(len(layers)):
            setattr(self, 'layer' + str(n), nn.Sequential(*layers[n]))

    def forward(self, input_x):
        """Patch Discriminator forward.

        Args:
            input_x (N x C x H1 x W2 tensor): Concatenation of images and
                semantic representations.
        Returns:
            (tuple):
              - output (N x 1 x H2 x W2 tensor): Discriminator output value.
                Before the sigmoid when using NSGAN.
              - features (list): lists of tensors of the intermediate
                activations.
        """
        res = [input_x]
        for n in range(self.num_layers + 2):
            layer = getattr(self, 'layer' + str(n))
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features


class WeightSharedMultiResPatchDiscriminator(nn.Module):
    """Multi-resolution patch discriminator with shared weights.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self, num_discriminators=3, kernel_size=3, num_image_channels=3, num_filters=64, num_layers=4, max_num_filters=512, activation_norm_type='', weight_norm_type='', **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn('Discriminator argument {} is not used'.format(key))
        self.num_discriminators = num_discriminators
        self.discriminator = NLayerPatchDiscriminator(kernel_size, num_image_channels, num_filters, num_layers, max_num_filters, activation_norm_type, weight_norm_type)
        None

    def forward(self, input_x):
        """Multi-resolution patch discriminator forward.

        Args:
            input_x (tensor) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_downsampled = input_x
        for i in range(self.num_discriminators):
            input_list.append(input_downsampled)
            output, features = self.discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(input_downsampled, scale_factor=0.5, mode='bilinear', align_corners=True)
        return output_list, features_list, input_list


class Discriminator(nn.Module):
    """UNIT discriminator. It can be either a multi-resolution patch
    discriminator like in the original implementation, or a
    global residual discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dis_cfg, data_cfg):
        super().__init__()
        if getattr(dis_cfg, 'patch_dis', True):
            self.discriminator_a = WeightSharedMultiResPatchDiscriminator(**vars(dis_cfg))
            self.discriminator_b = WeightSharedMultiResPatchDiscriminator(**vars(dis_cfg))
        else:
            self.discriminator_a = ResDiscriminator(**vars(dis_cfg))
            self.discriminator_b = ResDiscriminator(**vars(dis_cfg))

    def forward(self, data, net_G_output, gan_recon=False, real=True):
        """Returns the output of the discriminator.

        Args:
            data (dict):
              - images_a  (tensor) : Images in domain A.
              - images_b  (tensor) : Images in domain B.
            net_G_output (dict):
              - images_ab  (tensor) : Images translated from domain A to B by
                the generator.
              - images_ba  (tensor) : Images translated from domain B to A by
                the generator.
              - images_aa  (tensor) : Reconstructed images in domain A.
              - images_bb  (tensor) : Reconstructed images in domain B.
            gan_recon (bool): If ``True``, also classifies reconstructed images.
            real (bool): If ``True``, also classifies real images. Otherwise it
                only classifies generated images to save computation during the
                generator update.
        Returns:
            (dict):
              - out_ab (tensor): Output of the discriminator for images
                translated from domain A to B by the generator.
              - out_ab (tensor): Output of the discriminator for images
                translated from domain B to A by the generator.
              - fea_ab (tensor): Intermediate features of the discriminator
                for images translated from domain B to A by the generator.
              - fea_ba (tensor): Intermediate features of the discriminator
                for images translated from domain A to B by the generator.

              - out_a (tensor): Output of the discriminator for images
                in domain A.
              - out_b (tensor): Output of the discriminator for images
                in domain B.
              - fea_a (tensor): Intermediate features of the discriminator
                for images in domain A.
              - fea_b (tensor): Intermediate features of the discriminator
                for images in domain B.

              - out_aa (tensor): Output of the discriminator for
                reconstructed images in domain A.
              - out_bb (tensor): Output of the discriminator for
                reconstructed images in domain B.
              - fea_aa (tensor): Intermediate features of the discriminator
                for reconstructed images in domain A.
              - fea_bb (tensor): Intermediate features of the discriminator
                for reconstructed images in domain B.
        """
        out_ab, fea_ab, _ = self.discriminator_b(net_G_output['images_ab'])
        out_ba, fea_ba, _ = self.discriminator_a(net_G_output['images_ba'])
        output = dict(out_ba=out_ba, out_ab=out_ab, fea_ba=fea_ba, fea_ab=fea_ab)
        if real:
            out_a, fea_a, _ = self.discriminator_a(data['images_a'])
            out_b, fea_b, _ = self.discriminator_b(data['images_b'])
            output.update(dict(out_a=out_a, out_b=out_b, fea_a=fea_a, fea_b=fea_b))
        if gan_recon:
            out_aa, fea_aa, _ = self.discriminator_a(net_G_output['images_aa'])
            out_bb, fea_bb, _ = self.discriminator_b(net_G_output['images_bb'])
            output.update(dict(out_aa=out_aa, out_bb=out_bb, fea_aa=fea_aa, fea_bb=fea_bb))
        return output


class FPSEDiscriminator(nn.Module):

    def __init__(self, num_input_channels, num_labels, num_filters, kernel_size, weight_norm_type, activation_norm_type, do_multiscale, smooth_resample, no_label_except_largest_scale):
        super().__init__()
        self.do_multiscale = do_multiscale
        self.no_label_except_largest_scale = no_label_except_largest_scale
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        stride1_conv2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, stride=1, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity=nonlinearity, order='CNA')
        down_conv2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, stride=2, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity=nonlinearity, order='CNA')
        latent_conv2d_block = functools.partial(Conv2dBlock, kernel_size=1, stride=1, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity=nonlinearity, order='CNA')
        self.enc1 = down_conv2d_block(num_input_channels, num_filters)
        self.enc2 = down_conv2d_block(1 * num_filters, 2 * num_filters)
        self.enc3 = down_conv2d_block(2 * num_filters, 4 * num_filters)
        self.enc4 = down_conv2d_block(4 * num_filters, 8 * num_filters)
        self.enc5 = down_conv2d_block(8 * num_filters, 8 * num_filters)
        self.lat2 = latent_conv2d_block(2 * num_filters, 4 * num_filters)
        self.lat3 = latent_conv2d_block(4 * num_filters, 4 * num_filters)
        self.lat4 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.lat5 = latent_conv2d_block(8 * num_filters, 4 * num_filters)
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final2 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
        self.output = Conv2dBlock(num_filters * 2, num_labels + 1, kernel_size=1)
        if self.do_multiscale:
            self.final3 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
            self.final4 = stride1_conv2d_block(4 * num_filters, 2 * num_filters)
            if self.no_label_except_largest_scale:
                self.output3 = Conv2dBlock(num_filters * 2, 2, kernel_size=1)
                self.output4 = Conv2dBlock(num_filters * 2, 2, kernel_size=1)
            else:
                self.output3 = Conv2dBlock(num_filters * 2, num_labels + 1, kernel_size=1)
                self.output4 = Conv2dBlock(num_filters * 2, num_labels + 1, kernel_size=1)
        self.interpolator = functools.partial(F.interpolate, mode='nearest')
        if smooth_resample:
            self.interpolator = self.smooth_interp

    @staticmethod
    def smooth_interp(x, size):
        """Smooth interpolation of segmentation maps.

        Args:
            x (4D tensor): Segmentation maps.
            size(2D list): Target size (H, W).
        """
        x = F.interpolate(x, size=size, mode='area')
        onehot_idx = torch.argmax(x, dim=-3, keepdims=True)
        x.fill_(0.0)
        x.scatter_(1, onehot_idx, 1.0)
        return x

    def forward(self, images, segmaps, weights=None):
        feat11 = self.enc1(images)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        feat25 = self.lat5(feat15)
        feat24 = self.upsample2x(feat25) + self.lat4(feat14)
        feat23 = self.upsample2x(feat24) + self.lat3(feat13)
        feat22 = self.upsample2x(feat23) + self.lat2(feat12)
        feat32 = self.final2(feat22)
        results = []
        label_map = self.interpolator(segmaps, size=feat32.size()[2:])
        pred2 = self.output(feat32)
        features = [feat11, feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22]
        if weights is not None:
            label_map = label_map * weights[..., None, None]
        results.append({'pred': pred2, 'label': label_map})
        if self.do_multiscale:
            feat33 = self.final3(feat23)
            pred3 = self.output3(feat33)
            feat34 = self.final4(feat24)
            pred4 = self.output4(feat34)
            if self.no_label_except_largest_scale:
                label_map3 = torch.ones([pred3.size(0), 1, pred3.size(2), pred3.size(3)], device=pred3.device)
                label_map4 = torch.ones([pred4.size(0), 1, pred4.size(2), pred4.size(3)], device=pred4.device)
            else:
                label_map3 = self.interpolator(segmaps, size=pred3.size()[2:])
                label_map4 = self.interpolator(segmaps, size=pred4.size()[2:])
            if weights is not None:
                label_map3 = label_map3 * weights[..., None, None]
                label_map4 = label_map4 * weights[..., None, None]
            results.append({'pred': pred3, 'label': label_map3})
            results.append({'pred': pred4, 'label': label_map4})
        return results, features


class MultiPatchDiscriminator(nn.Module):
    """Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator part of the yaml config file.
        num_input_channels (int): Number of input channels.
    """

    def __init__(self, dis_cfg, num_input_channels):
        super(MultiPatchDiscriminator, self).__init__()
        kernel_size = getattr(dis_cfg, 'kernel_size', 4)
        num_filters = getattr(dis_cfg, 'num_filters', 64)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 3)
        num_layers = getattr(dis_cfg, 'num_layers', 3)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type', 'spectral_norm')
        self.nets_discriminator = []
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(kernel_size, num_input_channels, num_filters, num_layers, max_num_filters, activation_norm_type, weight_norm_type)
            self.add_module('discriminator_%d' % i, net_discriminator)
            self.nets_discriminator.append(net_discriminator)

    def forward(self, input_x):
        """Multi-resolution patch discriminator forward.

        Args:
            input_x (N x C x H x W tensor) : Concatenation of images and
                semantic representations.
        Returns:
            (dict):
              - output (list): list of output tensors produced by individual
                patch discriminators.
              - features (list): list of lists of features produced by
                individual patch discriminators.
        """
        output_list = []
        features_list = []
        input_downsampled = input_x
        for name, net_discriminator in self.named_children():
            if not name.startswith('discriminator_'):
                continue
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = F.interpolate(input_downsampled, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        output_x = dict()
        output_x['output'] = output_list
        output_x['features'] = features_list
        return output_x


class MultiResPatchDiscriminator(nn.Module):
    """Multi-resolution patch discriminator.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self, num_discriminators=3, kernel_size=3, num_image_channels=3, num_filters=64, num_layers=4, max_num_filters=512, activation_norm_type='', weight_norm_type='', **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn('Discriminator argument {} is not used'.format(key))
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(kernel_size, num_image_channels, num_filters, num_layers, max_num_filters, activation_norm_type, weight_norm_type)
            self.discriminators.append(net_discriminator)
        None

    def forward(self, input_x):
        """Multi-resolution patch discriminator forward.

        Args:
            input_x (tensor) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_downsampled = input_x
        for net_discriminator in self.discriminators:
            input_list.append(input_downsampled)
            output, features = net_discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(input_downsampled, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        return output_list, features_list, input_list


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


class ImageEncoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.model = encoder
        self.image_size = self.model.visual.input_resolution
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')

    @torch.no_grad()
    def forward(self, data, fake_images, align_corners=True):
        images = 0.5 * (1 + fake_images)
        images = F.interpolate(images, (self.image_size, self.image_size), mode='bicubic', align_corners=align_corners)
        images.clamp_(0, 1)
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        image_code = self.model.encode_image(images)
        return torch.cat((image_code, data['captions-clip']), dim=1)


def clean_resize(img_batch):
    batch_size = img_batch.size(0)
    img_batch = img_batch.cpu().numpy()
    fn_resize = build_resizer('clean')
    resized_batch = torch.zeros(batch_size, 3, 299, 299, device='cuda')
    for idx in range(batch_size):
        curr_img = img_batch[idx]
        img_np = curr_img.transpose((1, 2, 0))
        img_resize = fn_resize(img_np)
        resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)), device='cuda')
    resized_batch = resized_batch
    return resized_batch


class CleanInceptionV3(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = feature_extractor(name='torchscript_inception', resize_inside=False)

    def forward(self, img_batch, transform=True, **_kwargs):
        if transform:
            img_batch = torch.round(255 * (0.5 * img_batch + 0.5))
        resized_batch = clean_resize(img_batch)
        return self.model(resized_batch)


class NetLinLayer(nn.Module):
    """ A single linear layer used as placeholder for LPIPS learnt weights """

    def __init__(self, dim):
        super(NetLinLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, inp):
        out = self.weight * inp
        return out


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(in_feat, eps=1e-05):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


class LPNet(nn.Module):

    def __init__(self):
        super(LPNet, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.L = 5
        dims = [64, 128, 256, 512, 512]
        self.lins = nn.ModuleList([NetLinLayer(dims[i]) for i in range(self.L)])
        weights = torch.hub.load_state_dict_from_url('https://github.com/niopeng/CAM-Net/raw/main/code/models/weights/v0.1/vgg.pth')
        for i in range(self.L):
            self.lins[i].weight.data = torch.sqrt(weights['lin%d.model.1.weight' % i])

    def forward(self, in0, avg=False):
        in0 = 2 * in0 - 1
        in0_input = self.scaling_layer(in0)
        outs0 = self.net.forward(in0_input)
        feats0 = {}
        shapes = []
        res = []
        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])
        if avg:
            res = [self.lins[kk](feats0[kk]).mean([2, 3], keepdim=False) for kk in range(self.L)]
        else:
            for kk in range(self.L):
                cur_res = self.lins[kk](feats0[kk])
                shapes.append(cur_res.shape[-1])
                res.append(cur_res.reshape(cur_res.shape[0], -1))
        return res, shapes


class LPIPSNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = LPNet()

    @torch.no_grad()
    def forward(self, fake_images, fake_images_another, align_corners=True):
        features, shape = self._forward_single(fake_images)
        features_another, _ = self._forward_single(fake_images_another)
        result = 0
        for i, g_feat in enumerate(features):
            cur_diff = torch.sum((g_feat - features_another[i]) ** 2, dim=1) / shape[i] ** 2
            result += cur_diff
        return result

    def _forward_single(self, images):
        return self.model(torch.clamp(images, 0, 1))


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

    def forward(self, x):
        h = self.slice1(x)
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


class SwAV(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/swav', 'resnet50', pretrained=True)
        self.model.fc = torch.nn.Sequential()

    def forward(self, x, align_corners=True):
        y = self.model(F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=align_corners))
        return y


class Vgg16(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = vgg16(pretrained=True, init_weights=False)
        self.model.classifier = torch.nn.Sequential(*[self.model.classifier[i] for i in range(4)])

    def forward(self, x, align_corners=True):
        y = self.model(F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=align_corners))
        return y


class InceptionV3(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = inception_v3(transform_input=False, pretrained=True, init_weights=False)
        self.model.fc = torch.nn.Sequential()

    def forward(self, x, align_corners=True):
        y = self.model(F.interpolate(x, size=(299, 299), mode='bicubic', align_corners=align_corners))
        return y


class FIDInceptionA(inception.InceptionA):
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


class FIDInceptionC(inception.InceptionC):
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


class FIDInceptionE_1(inception.InceptionE):
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


class FIDInceptionE_2(inception.InceptionE):
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


class TFInceptionV3(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = inception_v3(transform_input=False, num_classes=1008, aux_logits=False, pretrained=False, init_weights=False)
        self.model.Mixed_5b = FIDInceptionA(192, pool_features=32)
        self.model.Mixed_5c = FIDInceptionA(256, pool_features=64)
        self.model.Mixed_5d = FIDInceptionA(288, pool_features=64)
        self.model.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        self.model.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        self.model.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        self.model.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        self.model.Mixed_7b = FIDInceptionE_1(1280)
        self.model.Mixed_7c = FIDInceptionE_2(2048)
        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.fc = torch.nn.Sequential()

    def forward(self, x, align_corners=True):
        y = self.model(F.interpolate(x, size=(299, 299), mode='bicubic', align_corners=align_corners))
        return y


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class Unet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=19, is_deconv=True, in_channels=3, is_batchnorm=True, image_size=512, use_dont_care=False):
        super(Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.image_size = image_size
        self.n_classes = n_classes
        self.use_dont_care = use_dont_care
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, images, align_corners=True):
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bicubic', align_corners=align_corners)
        conv1 = self.conv1(images)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        probs = self.final(up1)
        pred = torch.argmax(probs, dim=1)
        return pred


class DeepLabV2(nn.Module):

    def __init__(self, n_classes=182, image_size=512, use_dont_care=True):
        super(DeepLabV2, self).__init__()
        self.model = torch.hub.load('kazuto1011/deeplab-pytorch', 'deeplabv2_resnet101', pretrained=False, n_classes=182)
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.image_size = image_size
        self.mean = torch.tensor([104.008, 116.669, 122.675], device='cuda')
        self.n_classes = n_classes
        self.use_dont_care = use_dont_care

    def forward(self, images, align_corners=True):
        scale = self.image_size / max(images.shape[2:])
        images = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=align_corners)
        images = 255 * 0.5 * (images + 1)
        images = images.flip(1)
        images -= self.mean[None, :, None, None]
        _, _, H, W = images.shape
        logits = self.model(images)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=align_corners)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        return pred


def compute_hist(pred, gt, n_classes, use_dont_care):
    _, H, W = pred.size()
    gt = F.interpolate(gt.float(), (H, W), mode='nearest').long().squeeze(1)
    ignore_idx = n_classes if use_dont_care else -1
    all_hist = []
    for cur_pred, cur_gt in zip(pred, gt):
        keep = torch.logical_not(cur_gt == ignore_idx)
        merge = cur_pred[keep] * n_classes + cur_gt[keep]
        hist = torch.bincount(merge, minlength=n_classes ** 2)
        hist = hist.view((n_classes, n_classes))
        all_hist.append(hist)
    all_hist = torch.stack(all_hist)
    return all_hist


class SegmentationHistModel(nn.Module):

    def __init__(self, seg_network):
        super().__init__()
        self.seg_network = seg_network

    def forward(self, data, fake_images, align_corners=True):
        pred = self.seg_network(fake_images, align_corners=align_corners)
        gt = data['segmaps']
        gt = gt * 255.0
        gt = gt.long()
        return compute_hist(pred, gt, self.seg_network.n_classes, self.seg_network.use_dont_care)


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d


class SplatRenderer(object):
    """Splatting 3D point cloud into image using precomputed mapping."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the renderer."""
        self.seen_mask = None
        self.seen_time = None
        self.colors = None
        self.time_taken = 0
        self.call_idx = 0

    def num_points(self):
        """Number of points with assigned colors."""
        return np.sum(self.seen_mask)

    def _resize_arrays(self, max_point_idx):
        """Makes arrays bigger, if needed.
        Args:
            max_point_idx (int): Highest 3D point index seen so far.
        """
        if self.colors is None:
            old_max_point_idx = 0
        else:
            old_max_point_idx = self.colors.shape[0]
        if max_point_idx > old_max_point_idx:
            colors = np.zeros((max_point_idx, 3), dtype=np.uint8)
            seen_mask = np.zeros((max_point_idx, 1), dtype=np.uint8)
            seen_time = np.zeros((max_point_idx, 1), dtype=np.uint16)
            if old_max_point_idx > 0:
                colors[:old_max_point_idx] = self.colors
                seen_mask[:old_max_point_idx] = self.seen_mask
                seen_time[:old_max_point_idx] = self.seen_time
            self.colors = colors
            self.seen_mask = seen_mask
            self.seen_time = seen_time

    def update_point_cloud(self, image, point_info):
        """Updates point cloud with new points and colors.
        Args:
            image (H x W x 3, uint8): Select colors from this image to assign to
            3D points which do not have previously assigned colors.
            point_info (N x 3): (i, j, 3D point idx) per row containing
            mapping of image pixel to 3D point in point cloud.
        """
        if point_info is None or len(point_info) == 0:
            return
        start = time.time()
        self.call_idx += 1
        i_idxs = point_info[:, 0]
        j_idxs = point_info[:, 1]
        point_idxs = point_info[:, 2]
        max_point_idx = np.max(np.array(point_idxs)) + 1
        self._resize_arrays(max_point_idx)
        self.colors[point_idxs] = self.seen_mask[point_idxs] * self.colors[point_idxs] + (1 - self.seen_mask[point_idxs]) * image[i_idxs, j_idxs]
        self.seen_time[point_idxs] = self.seen_mask[point_idxs] * self.seen_time[point_idxs] + (1 - self.seen_mask[point_idxs]) * self.call_idx
        self.seen_mask[point_idxs] = 1
        end = time.time()
        self.time_taken += end - start

    def render_image(self, point_info, w, h, return_mask=False):
        """Creates image of (h, w) and fills in colors.
        Args:
            point_info (N x 3): (i, j, 3D point idx) per row containing
            mapping of image pixel to 3D point in point cloud.
            w (int): Width of output image.
            h (int): Height of output image.
            return_mask (bool): Return binary mask of coloring.
        Returns:
            (tuple):
              - output (H x W x 3, uint8): Image formed with mapping and colors.
              - mask (H x W x 1, uint8): Binary (255 or 0) mask of colorization.
        """
        output = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        if point_info is None or len(point_info) == 0:
            if return_mask:
                return output, mask
            else:
                return output
        start = time.time()
        i_idxs = point_info[:, 0]
        j_idxs = point_info[:, 1]
        point_idxs = point_info[:, 2]
        max_point_idx = np.max(np.array(point_idxs)) + 1
        self._resize_arrays(max_point_idx)
        output[i_idxs, j_idxs] = self.colors[point_idxs]
        end = time.time()
        self.time_taken += end - start
        if return_mask:
            mask[i_idxs, j_idxs] = 255 * self.seen_mask[point_idxs]
            return output, mask
        else:
            return output


def _calculate_model_size(model):
    """Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Fromage(Optimizer):
    """Fromage optimizer implementation (https://arxiv.org/abs/2002.03432)"""

    def __init__(self, params, lr=required, momentum=0):
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        defaults = dict(lr=lr, momentum=momentum)
        super(Fromage, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()
                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(-group['lr'], d_p * (p_norm / d_p_norm))
                else:
                    p.data.add_(-group['lr'], d_p)
                p.data /= math.sqrt(1 + group['lr'] ** 2)
        return loss


class Madam(Optimizer):
    """MADAM optimizer implementation (https://arxiv.org/abs/2006.14560)"""

    def __init__(self, params, lr=required, scale=3.0, g_bound=None, momentum=0):
        self.scale = scale
        self.g_bound = g_bound
        defaults = dict(lr=lr, momentum=momentum)
        super(Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['max'] = self.scale * (p * p).mean().sqrt().item()
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                bias_correction = 1 - 0.999 ** state['step']
                state['exp_avg_sq'] = 0.999 * state['exp_avg_sq'] + 0.001 * p.grad.data ** 2
                g_normed = p.grad.data / (state['exp_avg_sq'] / bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                if self.g_bound is not None:
                    g_normed.clamp_(-self.g_bound, self.g_bound)
                p.data *= torch.exp(-group['lr'] * g_normed * torch.sign(p.data))
                p.data.clamp_(-state['max'], state['max'])
        return loss


def get_optimizer(cfg_opt, net):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        net (obj): PyTorch network object.

    Returns:
        (obj): Pytorch optimizer
    """
    if hasattr(net, 'get_param_groups'):
        params = net.get_param_groups(cfg_opt)
    else:
        params = net.parameters()
    return get_optimizer_for_params(cfg_opt, params)


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.lr_policy.type == 'step':
        scheduler = lr_scheduler.StepLR(opt, step_size=cfg_opt.lr_policy.step_size, gamma=cfg_opt.lr_policy.gamma)
    elif cfg_opt.lr_policy.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)
    elif cfg_opt.lr_policy.type == 'linear':
        decay_start = cfg_opt.lr_policy.decay_start
        decay_end = cfg_opt.lr_policy.decay_end
        decay_target = cfg_opt.lr_policy.decay_target

        def sch(x):
            return min(max(((x - decay_start) * decay_target + decay_end - x) / (decay_end - decay_start), decay_target), 1.0)
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.format(cfg_opt.lr_policy.type))
    return scheduler


def get_rank():
    """Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def set_random_seed(seed, by_rank=False):
    """Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    if by_rank:
        seed += get_rank()
    None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(init_type='normal', gain=0.02, bias=None):
    """Initialize weights in the network.

    Args:
        init_type (str): The name of the initialization scheme.
        gain (float): The parameter that is required for the initialization
            scheme.
        bias (object): If not ``None``, specifies the initialization parameter
            for bias.

    Returns:
        (obj): init function to be applied.
    """

    def init_func(m):
        """Init function

        Args:
            m: module to be weight initialized.
        """
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1 or class_name.find('Embedding') != -1):
            lr_mul = getattr(m, 'lr_mul', 1.0)
            gain_final = gain / lr_mul
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain_final)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain_final)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain_final)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                with torch.no_grad():
                    m.weight.data *= gain_final
            elif init_type == 'kaiming_linear':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='linear')
                with torch.no_grad():
                    m.weight.data *= gain_final
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain_final)
            elif init_type == 'none':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            if init_type == 'none':
                pass
            elif bias is not None:
                bias_type = getattr(bias, 'type', 'normal')
                if bias_type == 'normal':
                    bias_gain = getattr(bias, 'gain', 0.5)
                    init.normal_(m.bias.data, 0.0, bias_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % bias_type)
            else:
                init.constant_(m.bias.data, 0.0)
    return init_func


def weights_rescale():

    def init_func(m):
        if hasattr(m, 'init_gain'):
            for name, p in m.named_parameters():
                if 'output_scale' not in name:
                    p.data.mul_(m.init_gain)
    return init_func


class ScaledLR(object):

    def __init__(self, weight_name, bias_name):
        self.weight_name = weight_name
        self.bias_name = bias_name

    def compute_weight(self, module):
        weight = getattr(module, self.weight_name + '_ori')
        return weight * module.weight_scale

    def compute_bias(self, module):
        bias = getattr(module, self.bias_name + '_ori')
        if bias is not None:
            return bias * module.bias_scale
        else:
            return None

    @staticmethod
    def apply(module, weight_name, bias_name, lr_mul, equalized):
        assert weight_name == 'weight'
        assert bias_name == 'bias'
        fn = ScaledLR(weight_name, bias_name)
        module.register_forward_pre_hook(fn)
        if hasattr(module, bias_name):
            bias = getattr(module, bias_name)
            delattr(module, bias_name)
            module.register_parameter(bias_name + '_ori', bias)
        else:
            bias = None
            setattr(module, bias_name + '_ori', bias)
        if bias is not None:
            setattr(module, bias_name, bias.data)
        else:
            setattr(module, bias_name, None)
        module.register_buffer('bias_scale', torch.tensor(lr_mul))
        if hasattr(module, weight_name + '_orig'):
            weight = getattr(module, weight_name + '_orig')
            delattr(module, weight_name + '_orig')
            module.register_parameter(weight_name + '_ori', weight)
            setattr(module, weight_name + '_orig', weight.data)
            module._forward_pre_hooks = collections.OrderedDict(reversed(list(module._forward_pre_hooks.items())))
            module.use_sn = True
        else:
            weight = getattr(module, weight_name)
            delattr(module, weight_name)
            module.register_parameter(weight_name + '_ori', weight)
            setattr(module, weight_name, weight.data)
            module.use_sn = False
        if equalized:
            fan_in = weight.data.size(1) * weight.data[0][0].numel()
            module.register_buffer('weight_scale', torch.tensor(lr_mul * (1 / fan_in) ** 0.5))
        else:
            module.register_buffer('weight_scale', torch.tensor(lr_mul))
        module.lr_mul = module.weight_scale
        module.base_lr_mul = lr_mul
        return fn

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module)
        delattr(module, self.weight_name + '_ori')
        if module.use_sn:
            setattr(module, self.weight_name + '_orig', weight.detach())
        else:
            delattr(module, self.weight_name)
            module.register_parameter(self.weight_name, torch.nn.Parameter(weight.detach()))
        with torch.no_grad():
            bias = self.compute_bias(module)
        delattr(module, self.bias_name)
        delattr(module, self.bias_name + '_ori')
        if bias is not None:
            module.register_parameter(self.bias_name, torch.nn.Parameter(bias.detach()))
        else:
            module.register_parameter(self.bias_name, None)
        module.lr_mul = 1.0
        module.base_lr_mul = 1.0

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        if module.use_sn:
            setattr(module, self.weight_name + '_orig', weight)
        else:
            setattr(module, self.weight_name, weight)
        bias = self.compute_bias(module)
        setattr(module, self.bias_name, bias)


def remove_weight_norms(module, weight_name='weight', bias_name='bias'):
    if hasattr(module, 'weight_ori') or hasattr(module, 'weight_orig'):
        for k in list(module._forward_pre_hooks.keys()):
            hook = module._forward_pre_hooks[k]
            if isinstance(hook, ScaledLR) or isinstance(hook, SpectralNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
        for k, hook in module._state_dict_hooks.items():
            if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == weight_name:
                del module._state_dict_hooks[k]
                break
        for k, hook in module._load_state_dict_pre_hooks.items():
            if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == weight_name:
                del module._load_state_dict_pre_hooks[k]
                break
    return module


def requires_grad(model, require=True):
    """ Set a model to require gradient or not.

    Args:
        model (nn.Module): Neural network model.
        require (bool): Whether the network requires gradient or not.

    Returns:

    """
    for p in model.parameters():
        p.requires_grad = require


class ModelAverage(nn.Module):
    """In this model average implementation, the spectral layers are
    absorbed in the model parameter by default. If such options are
    turned on, be careful with how you do the training. Remember to
    re-estimate the batch norm parameters before using the model.

    Args:
        module (torch nn module): Torch network.
        beta (float): Moving average weights. How much we weight the past.
        start_iteration (int): From which iteration, we start the update.
        remove_sn (bool): Whether we remove the spectral norm when we it.
    """

    def __init__(self, module, beta=0.9999, start_iteration=1000, remove_wn_wrapper=True):
        super(ModelAverage, self).__init__()
        self.module = module
        self.averaged_model = copy.deepcopy(self.module)
        self.beta = beta
        self.remove_wn_wrapper = remove_wn_wrapper
        self.start_iteration = start_iteration
        self.register_buffer('num_updates_tracked', torch.tensor(0, dtype=torch.long))
        self.num_updates_tracked = self.num_updates_tracked
        if self.remove_wn_wrapper:
            self.copy_s2t()
            self.averaged_model.apply(remove_weight_norms)
            self.dim = 0
        else:
            self.averaged_model.eval()
        requires_grad(self.averaged_model, False)

    def forward(self, *inputs, **kwargs):
        """PyTorch module forward function overload."""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def update_average(self):
        """Update the moving average."""
        self.num_updates_tracked += 1
        if self.num_updates_tracked <= self.start_iteration:
            beta = 0.0
        else:
            beta = self.beta
        source_dict = self.module.state_dict()
        target_dict = self.averaged_model.state_dict()
        for key in target_dict:
            if 'num_batches_tracked' in key:
                continue
            if self.remove_wn_wrapper:
                if key.endswith('weight'):
                    if key + '_ori' in source_dict:
                        source_param = source_dict[key + '_ori'] * source_dict[key + '_scale']
                    elif key + '_orig' in source_dict:
                        source_param = source_dict[key + '_orig']
                    elif key in source_dict:
                        source_param = source_dict[key]
                    else:
                        raise ValueError(f'{key} required in the averaged model but not found in the regular model.')
                    source_param = source_param.detach()
                    if key + '_orig' in source_dict:
                        source_param = self.sn_compute_weight(source_param, source_dict[key + '_u'], source_dict[key + '_v'])
                elif key.endswith('bias') and key + '_ori' in source_dict:
                    source_param = source_dict[key + '_ori'] * source_dict[key + '_scale']
                else:
                    source_param = source_dict[key]
                target_dict[key].data.mul_(beta).add_(source_param.data, alpha=1 - beta)
            else:
                target_dict[key].data.mul_(beta).add_(source_dict[key].data, alpha=1 - beta)

    @torch.no_grad()
    def copy_t2s(self):
        """Copy the original weights to the moving average weights."""
        target_dict = self.module.state_dict()
        source_dict = self.averaged_model.state_dict()
        beta = 0.0
        for key in source_dict:
            target_dict[key].data.copy_(target_dict[key].data * beta + source_dict[key].data * (1 - beta))

    @torch.no_grad()
    def copy_s2t(self):
        """ Copy state_dictionary from source to target.
        Here source is the regular module and the target is the moving
        average module. Basically, we will copy weights in the regular module
        to the moving average module.
        """
        source_dict = self.module.state_dict()
        target_dict = self.averaged_model.state_dict()
        beta = 0.0
        for key in source_dict:
            target_dict[key].data.copy_(target_dict[key].data * beta + source_dict[key].data * (1 - beta))

    def __repr__(self):
        """Returns a string that holds a printable representation of an
        object"""
        return self.module.__repr__()

    def sn_reshape_weight_to_matrix(self, weight):
        """Reshape weight to obtain the matrix form.

        Args:
            weight (Parameters): pytorch layer parameter tensor.

        Returns:
            (Parameters): Reshaped weight matrix
        """
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def sn_compute_weight(self, weight, u, v):
        """Compute the spectral norm normalized matrix.

        Args:
            weight (Parameters): pytorch layer parameter tensor.
            u (tensor): left singular vectors.
            v (tensor) right singular vectors

        Returns:
            (Parameters): weight parameter object.
        """
        weight_mat = self.sn_reshape_weight_to_matrix(weight)
        sigma = torch.sum(u * torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight


class WrappedModel(nn.Module):
    """Dummy wrapping the module.
    """

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """PyTorch module forward function overload."""
        return self.module(*args, **kwargs)


def _wrap_model(cfg, model):
    """Wrap a model for distributed data parallel training.

    Args:
        model (obj): PyTorch network model.

    Returns:
        (obj): Wrapped PyTorch network model.
    """
    if torch.distributed.is_available() and dist.is_initialized():
        find_unused_parameters = cfg.trainer.distributed_data_parallel_params.find_unused_parameters
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=find_unused_parameters, broadcast_buffers=False)
    else:
        return WrappedModel(model)


def get_world_size():
    """Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def wrap_model_and_optimizer(cfg, net_G, net_D, opt_G, opt_D):
    """Wrap the networks and the optimizers with AMP DDP and (optionally)
    model average.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network object.
        net_D (obj): Discriminator network object.
        opt_G (obj): Generator optimizer object.
        opt_D (obj): Discriminator optimizer object.

    Returns:
        (dict):
          - net_G (obj): Generator network object.
          - net_D (obj): Discriminator network object.
          - opt_G (obj): Generator optimizer object.
          - opt_D (obj): Discriminator optimizer object.
    """
    if cfg.trainer.model_average_config.enabled:
        if hasattr(cfg.trainer.model_average_config, 'g_smooth_img'):
            cfg.trainer.model_average_config.beta = 0.5 ** (cfg.data.train.batch_size * get_world_size() / cfg.trainer.model_average_config.g_smooth_img)
            None
        net_G = ModelAverage(net_G, cfg.trainer.model_average_config.beta, cfg.trainer.model_average_config.start_iteration, cfg.trainer.model_average_config.remove_sn)
    if cfg.trainer.model_average_config.enabled:
        net_G_module = net_G.module
    else:
        net_G_module = net_G
    if hasattr(net_G_module, 'custom_init'):
        net_G_module.custom_init()
    net_G = _wrap_model(cfg, net_G)
    net_D = _wrap_model(cfg, net_D)
    return net_G, net_D, opt_G, opt_D


def get_model_optimizer_and_scheduler(cfg, seed=0):
    """Return the networks, the optimizers, and the schedulers. We will
    first set the random seed to a fixed value so that each GPU copy will be
    initialized to have the same network weights. We will then use different
    random seeds for different GPUs. After this we will wrap the generator
    with a moving average model if applicable. It is followed by getting the
    optimizers and data distributed data parallel wrapping.

    Args:
        cfg (obj): Global configuration.
        seed (int): Random seed.

    Returns:
        (dict):
          - net_G (obj): Generator network object.
          - net_D (obj): Discriminator network object.
          - opt_G (obj): Generator optimizer object.
          - opt_D (obj): Discriminator optimizer object.
          - sch_G (obj): Generator optimizer scheduler object.
          - sch_D (obj): Discriminator optimizer scheduler object.
    """
    set_random_seed(seed, by_rank=False)
    lib_G = importlib.import_module(cfg.gen.type)
    lib_D = importlib.import_module(cfg.dis.type)
    net_G = lib_G.Generator(cfg.gen, cfg.data)
    net_D = lib_D.Discriminator(cfg.dis, cfg.data)
    None
    init_bias = getattr(cfg.trainer.init, 'bias', None)
    net_G.apply(weights_init(cfg.trainer.init.type, cfg.trainer.init.gain, init_bias))
    net_D.apply(weights_init(cfg.trainer.init.type, cfg.trainer.init.gain, init_bias))
    net_G.apply(weights_rescale())
    net_D.apply(weights_rescale())
    net_G = net_G
    net_D = net_D
    set_random_seed(seed, by_rank=True)
    None
    None
    opt_G = get_optimizer(cfg.gen_opt, net_G)
    opt_D = get_optimizer(cfg.dis_opt, net_D)
    net_G, net_D, opt_G, opt_D = wrap_model_and_optimizer(cfg, net_G, net_D, opt_G, opt_D)
    sch_G = get_scheduler(cfg.gen_opt, opt_G)
    sch_D = get_scheduler(cfg.dis_opt, opt_D)
    return net_G, net_D, opt_G, opt_D, sch_G, sch_D


def get_trainer(cfg, net_G, net_D=None, opt_G=None, opt_D=None, sch_G=None, sch_D=None, train_data_loader=None, val_data_loader=None):
    """Return the trainer object.

    Args:
        cfg (Config): Loaded config object.
        net_G (obj): Generator network object.
        net_D (obj): Discriminator network object.
        opt_G (obj): Generator optimizer object.
        opt_D (obj): Discriminator optimizer object.
        sch_G (obj): Generator optimizer scheduler object.
        sch_D (obj): Discriminator optimizer scheduler object.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.

    Returns:
        (obj): Trainer object.
    """
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, train_data_loader, val_data_loader)
    return trainer


def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    """Get a grid ranging [-1, 1] of 2D/3D coordinates.

    Args:
        batchsize (int) : Batch size.
        size (tuple) : (height, width) or (depth, height, width).
        minval (float) : minimum value in returned grid.
        maxval (float) : maximum value in returned grid.
    Returns:
        t_grid (4D tensor) : Grid of coordinates.
    """
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError('Dimension can only be 2 or 3.')
    x = torch.linspace(minval, maxval, cols)
    x = x.view(1, 1, 1, cols)
    x = x.expand(batchsize, 1, rows, cols)
    y = torch.linspace(minval, maxval, rows)
    y = y.view(1, 1, rows, 1)
    y = y.expand(batchsize, 1, rows, cols)
    t_grid = torch.cat([x, y], dim=1)
    if len(size) == 3:
        z = torch.linspace(minval, maxval, deps)
        z = z.view(1, 1, deps, 1, 1)
        z = z.expand(batchsize, 1, deps, rows, cols)
        t_grid = t_grid.unsqueeze(2).expand(batchsize, 2, deps, rows, cols)
        t_grid = torch.cat([t_grid, z], dim=1)
    t_grid.requires_grad = False
    return t_grid


def resample(image, flow):
    """Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor) : Image to resample.
        flow (Nx2xHxW tensor) : Optical flow to resample the image.
    Returns:
        output (NxCxHxW tensor) : Resampled image.
    """
    assert flow.shape[1] == 2
    b, c, h, w = image.size()
    grid = get_grid(b, (h, w))
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    try:
        output = F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    except Exception:
        output = F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
    return output


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, three_channel_output=True):
    """Convert tensor to image.

    Args:
        image_tensor (torch.tensor or list of torch.tensor): If tensor then
            (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.
        normalize (bool): Is the input image normalized or not?
            three_channel_output (bool): Should single channel images be made 3
            channel in output?

    Returns:
        (numpy.ndarray, list if case 1, 2 above).
    """
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x, imtype, normalize) for x in image_tensor]
    if image_tensor.dim() == 5 or image_tensor.dim() == 4:
        return [tensor2im(image_tensor[idx], imtype, normalize) for idx in range(image_tensor.size(0))]
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 and three_channel_output:
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        elif image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:, :, :3]
        return image_numpy.astype(imtype)


class ContentEncoder(nn.Module):
    """Improved UNIT encoder. The network consists of:

    - input layers
    - $(num_downsamples) convolutional blocks
    - $(num_res_blocks) residual blocks.
    - output layer.

    Args:
        num_downsamples (int): Number of times we reduce
            resolution by 2x2.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_image_channels (int): Number of input image channels.
        num_filters (int): Base filter numbers.
        max_num_filters (int): Maximum number of filters in the encoder.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
        nonlinearity (str): Type of nonlinear activation function.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
    """

    def __init__(self, num_downsamples, num_res_blocks, num_image_channels, num_filters, max_num_filters, padding_mode, activation_norm_type, weight_norm_type, nonlinearity, pre_act=False):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode, activation_norm_type=activation_norm_type, weight_norm_type=weight_norm_type, nonlinearity=nonlinearity)
        if not pre_act or activation_norm_type != '' and activation_norm_type != 'none':
            conv_params['inplace_nonlinearity'] = True
        order = 'pre_act' if pre_act else 'CNACNA'
        model = []
        model += [Conv2dBlock(num_image_channels, num_filters, 7, 1, 3, **conv_params)]
        for i in range(num_downsamples):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Conv2dBlock(num_filters_prev, num_filters, 4, 2, 1, **conv_params)]
        for _ in range(num_res_blocks):
            model += [Res2dBlock(num_filters, num_filters, **conv_params, order=order)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        """

        Args:
            x (tensor): Input image.
        """
        return self.model(x)


class Decoder(nn.Module):
    """Improved UNIT decoder. The network consists of:

    - $(num_res_blocks) residual blocks.
    - $(num_upsamples) residual blocks or convolutional blocks
    - output layer.

    Args:
        num_upsamples (int): Number of times we increase resolution by 2x2.
        num_res_blocks (int): Number of residual blocks.
        num_filters (int): Base filter numbers.
        num_image_channels (int): Number of input image channels.
        padding_mode (string): Type of padding.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
        nonlinearity (str): Type of nonlinear activation function.
        output_nonlinearity (str): Type of nonlinearity before final output,
            ``'tanh'`` or ``'none'``.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
        apply_noise (bool): If ``True``, injects Gaussian noise.
    """

    def __init__(self, num_upsamples, num_res_blocks, num_filters, num_image_channels, padding_mode, activation_norm_type, weight_norm_type, nonlinearity, output_nonlinearity, pre_act=False, apply_noise=False):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode, nonlinearity=nonlinearity, inplace_nonlinearity=True, apply_noise=apply_noise, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type)
        order = 'pre_act' if pre_act else 'CNACNA'
        self.decoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.decoder += [Res2dBlock(num_filters, num_filters, **conv_params, order=order)]
        for i in range(num_upsamples):
            self.decoder += [NearestUpsample(scale_factor=2)]
            self.decoder += [Conv2dBlock(num_filters, num_filters // 2, 5, 1, 2, **conv_params)]
            num_filters //= 2
        self.decoder += [Conv2dBlock(num_filters, num_image_channels, 7, 1, 3, nonlinearity=output_nonlinearity, padding_mode=padding_mode)]

    def forward(self, x):
        """

        Args:
            x (tensor): Content embedding of the content image.
        """
        for block in self.decoder:
            x = block(x)
        return x


class LinearBlock(_BaseConvBlock):
    """A Wrapper class that wraps ``torch.nn.Linear`` with normalization and
    nonlinearity.

    Args:
        in_features (int): Number of channels in the input tensor.
        out_features (int): Number of channels in the output tensor.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_features, out_features, bias=True, weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, order='CNA', clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0, **_kwargs):
        if bool(_kwargs):
            warnings.warn(f'Unused keyword arguments {_kwargs}')
        super().__init__(in_features, out_features, None, None, None, None, None, bias, None, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, False, order, 0, clamp, blur_kernel, output_scale, init_gain)


class MLP(nn.Module):
    """The multi-layer perceptron (MLP) that maps Gaussian style code to a
    feature vector that is given as the conditional input to AdaIN.

    Args:
        input_dim (int): Number of channels in the input tensor.
        output_dim (int): Number of channels in the output tensor.
        latent_dim (int): Number of channels in the latent features.
        num_layers (int): Number of layers in the MLP.
        norm (str): Type of activation normalization.
        nonlinearity (str): Type of nonlinear activation function.
    """

    def __init__(self, input_dim, output_dim, latent_dim, num_layers, norm, nonlinearity):
        super().__init__()
        model = []
        model += [LinearBlock(input_dim, latent_dim, activation_norm_type=norm, nonlinearity=nonlinearity)]
        for i in range(num_layers - 2):
            model += [LinearBlock(latent_dim, latent_dim, activation_norm_type=norm, nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_dim, output_dim, activation_norm_type=norm, nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """

        Args:
            x (tensor): Input image.
        """
        return self.model(x.view(x.size(0), -1))


class StyleEncoder(nn.Module):
    """Style Encode constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """

    def __init__(self, style_enc_cfg):
        super(StyleEncoder, self).__init__()
        input_image_channels = style_enc_cfg.input_image_channels
        num_filters = style_enc_cfg.num_filters
        kernel_size = style_enc_cfg.kernel_size
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        style_dims = style_enc_cfg.style_dims
        weight_norm_type = style_enc_cfg.weight_norm_type
        activation_norm_type = 'none'
        nonlinearity = 'leakyrelu'
        base_conv2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, stride=2, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity=nonlinearity)
        self.layer1 = base_conv2d_block(input_image_channels, num_filters)
        self.layer2 = base_conv2d_block(num_filters * 1, num_filters * 2)
        self.layer3 = base_conv2d_block(num_filters * 2, num_filters * 4)
        self.layer4 = base_conv2d_block(num_filters * 4, num_filters * 8)
        self.layer5 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.layer6 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.fc_mu = LinearBlock(num_filters * 8 * 4 * 4, style_dims)
        self.fc_var = LinearBlock(num_filters * 8 * 4 * 4, style_dims)

    def forward(self, input_x):
        """SPADE Style Encoder forward.

        Args:
            input_x (N x 3 x H x W tensor): input images.
        Returns:
            (tuple):
              - mu (N x C tensor): Mean vectors.
              - logvar (N x C tensor): Log-variance vectors.
              - z (N x C tensor): Style code vectors.
        """
        if input_x.size(2) != 256 or input_x.size(3) != 256:
            input_x = F.interpolate(input_x, size=(256, 256), mode='bilinear')
        x = self.layer1(input_x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return mu, logvar, z


class COCOFUNITTranslator(nn.Module):
    """COCO-FUNIT Generator architecture.

    Args:
        num_filters (int): Base filter numbers.
        num_filters_mlp (int): Base filter number in the MLP module.
        style_dims (int): Dimension of the style code.
        usb_dims (int): Dimension of the universal style bias code.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_mlp_blocks (int): Number of layers in the MLP module.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
        num_image_channels (int): Number of input image channels.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
    """

    def __init__(self, num_filters=64, num_filters_mlp=256, style_dims=64, usb_dims=1024, num_res_blocks=2, num_mlp_blocks=3, num_downsamples_style=4, num_downsamples_content=2, num_image_channels=3, weight_norm_type='', **kwargs):
        super().__init__()
        self.style_encoder = StyleEncoder(num_downsamples_style, num_image_channels, num_filters, style_dims, 'reflect', 'none', weight_norm_type, 'relu')
        self.content_encoder = ContentEncoder(num_downsamples_content, num_res_blocks, num_image_channels, num_filters, 'reflect', 'instance', weight_norm_type, 'relu')
        self.decoder = Decoder(self.content_encoder.output_dim, num_filters_mlp, num_image_channels, num_downsamples_content, 'reflect', weight_norm_type, 'relu')
        self.usb = torch.nn.Parameter(torch.randn(1, usb_dims))
        self.mlp = MLP(style_dims, num_filters_mlp, num_filters_mlp, num_mlp_blocks, 'none', 'relu')
        num_content_mlp_blocks = 2
        num_style_mlp_blocks = 2
        self.mlp_content = MLP(self.content_encoder.output_dim, style_dims, num_filters_mlp, num_content_mlp_blocks, 'none', 'relu')
        self.mlp_style = MLP(style_dims + usb_dims, style_dims, num_filters_mlp, num_style_mlp_blocks, 'none', 'relu')

    def forward(self, images):
        """Reconstruct the input image by combining the computer content and
        style code.

        Args:
            images (tensor): Input image tensor.
        """
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        """Encoder images to get their content and style codes.

        Args:
            images (tensor): Input image tensor.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        """Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images


class AttentionModule(nn.Module):
    """Attention module constructor.

    Args:
       atn_cfg (obj): Generator definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file
       conv_2d_block: Conv2DBlock constructor.
       num_filters_each_layer (int): The number of filters in each layer.
    """

    def __init__(self, atn_cfg, data_cfg, conv_2d_block, num_filters_each_layer):
        super().__init__()
        self.initial_few_shot_K = data_cfg.initial_few_shot_K
        num_input_channels = data_cfg.num_input_channels
        num_filters = getattr(atn_cfg, 'num_filters', 32)
        self.num_downsample_atn = getattr(atn_cfg, 'num_downsamples', 2)
        self.atn_query_first = conv_2d_block(num_input_channels, num_filters)
        self.atn_key_first = conv_2d_block(num_input_channels, num_filters)
        for i in range(self.num_downsamples_atn):
            f_in, f_out = num_filters_each_layer[i], num_filters_each_layer[i + 1]
            setattr(self, 'atn_key_%d' % i, conv_2d_block(f_in, f_out, stride=2))
            setattr(self, 'atn_query_%d' % i, conv_2d_block(f_in, f_out, stride=2))

    def forward(self, in_features, label, ref_label, attention=None):
        """Get the attention map to combine multiple image features in the
        case of multiple reference images.

        Args:
            in_features ((NxK)xC1xH1xW1 tensor): Input feaures.
            label (NxC2xH2xW2 tensor): Target label.
            ref_label (NxC2xH2xW2 tensor): Reference label.
            attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
        Returns:
            (tuple):
              - out_features (NxC1xH1xW1 tensor): Attention-combined features.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention scores.
        """
        b, c, h, w = in_features.size()
        k = self.initial_few_shot_K
        b = b // k
        if attention is None:
            atn_key = self.attention_encode(ref_label, 'atn_key')
            atn_query = self.attention_encode(label, 'atn_query')
            atn_key = atn_key.view(b, k, c, -1).permute(0, 1, 3, 2).contiguous().view(b, -1, c)
            atn_query = atn_query.view(b, c, -1)
            energy = torch.bmm(atn_key, atn_query)
            attention = nn.Softmax(dim=1)(energy)
        in_features = in_features.view(b, k, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        out_features = torch.bmm(in_features, attention).view(b, c, h, w)
        atn_vis = attention.view(b, k, h * w, h * w).sum(2).view(b, k, h, w)
        return out_features, attention, atn_vis[-1:, 0:1]

    def attention_encode(self, img, net_name):
        """Encode the input image to get the attention map.

        Args:
            img (NxCxHxW tensor): Input image.
            net_name (str): Name for attention network.
        Returns:
            x (NxC2xH2xW2 tensor): Encoded feature.
        """
        x = getattr(self, net_name + '_first')(img)
        for i in range(self.num_downsample_atn):
            x = getattr(self, net_name + '_' + str(i))(x)
        return x


class _BaseHyperConvBlock(_BaseConvBlock):
    """An abstract wrapper class that wraps a hyper convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, is_hyper_conv, is_hyper_norm, order, input_dim, clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        self.is_hyper_conv = is_hyper_conv
        if is_hyper_conv:
            weight_norm_type = 'none'
        if is_hyper_norm:
            activation_norm_type = 'hyper_' + activation_norm_type
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, input_dim, clamp, blur_kernel, output_scale, init_gain)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        if input_dim == 0:
            raise ValueError('HyperLinearBlock is not supported.')
        else:
            name = 'HyperConv' if self.is_hyper_conv else 'nn.Conv'
            layer_type = eval(name + '%dd' % input_dim)
            layer = layer_type(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        return layer


class HyperConv2dBlock(_BaseHyperConvBlock):
    """A Wrapper class that wraps ``HyperConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, is_hyper_conv=False, is_hyper_norm=False, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, blur=False, order='CNA', clamp=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, is_hyper_conv, is_hyper_norm, order, 2, clamp)


class LabelEmbedder(nn.Module):
    """Embed the input label map to get embedded features.

    Args:
        emb_cfg (obj): Embed network configuration.
        num_input_channels (int): Number of input channels.
        num_hyper_layers (int): Number of hyper layers.
    """

    def __init__(self, emb_cfg, num_input_channels, num_hyper_layers=0):
        super().__init__()
        num_filters = getattr(emb_cfg, 'num_filters', 32)
        max_num_filters = getattr(emb_cfg, 'max_num_filters', 1024)
        self.arch = getattr(emb_cfg, 'arch', 'encoderdecoder')
        self.num_downsamples = num_downsamples = getattr(emb_cfg, 'num_downsamples', 5)
        kernel_size = getattr(emb_cfg, 'kernel_size', 3)
        weight_norm_type = getattr(emb_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = getattr(emb_cfg, 'activation_norm_type', 'none')
        self.unet = 'unet' in self.arch
        self.has_decoder = 'decoder' in self.arch or self.unet
        self.num_hyper_layers = num_hyper_layers if num_hyper_layers != -1 else num_downsamples
        base_conv_block = partial(HyperConv2dBlock, kernel_size=kernel_size, padding=kernel_size // 2, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='leakyrelu')
        ch = [min(max_num_filters, num_filters * 2 ** i) for i in range(num_downsamples + 1)]
        self.conv_first = base_conv_block(num_input_channels, num_filters, activation_norm_type='none')
        for i in range(num_downsamples):
            is_hyper_conv = i < num_hyper_layers and not self.has_decoder
            setattr(self, 'down_%d' % i, base_conv_block(ch[i], ch[i + 1], stride=2, is_hyper_conv=is_hyper_conv))
        if self.has_decoder:
            self.upsample = nn.Upsample(scale_factor=2)
            for i in reversed(range(num_downsamples)):
                ch_i = ch[i + 1] * (2 if self.unet and i != num_downsamples - 1 else 1)
                setattr(self, 'up_%d' % i, base_conv_block(ch_i, ch[i], is_hyper_conv=i < num_hyper_layers))

    def forward(self, input, weights=None):
        """Embedding network forward.

        Args:
            input (NxCxHxW tensor): Network input.
            weights (list of tensors): Conv weights if using hyper network.
        Returns:
            output (list of tensors): Network outputs at different layers.
        """
        if input is None:
            return None
        output = [self.conv_first(input)]
        for i in range(self.num_downsamples):
            layer = getattr(self, 'down_%d' % i)
            if i >= self.num_hyper_layers or self.has_decoder:
                conv = layer(output[-1])
            else:
                conv = layer(output[-1], conv_weights=weights[i])
            output.append(conv)
        if not self.has_decoder:
            return output
        if not self.unet:
            output = [output[-1]]
        for i in reversed(range(self.num_downsamples)):
            input_i = output[-1]
            if self.unet and i != self.num_downsamples - 1:
                input_i = torch.cat([input_i, output[i + 1]], dim=1)
            input_i = self.upsample(input_i)
            layer = getattr(self, 'up_%d' % i)
            if i >= self.num_hyper_layers:
                conv = layer(input_i)
            else:
                conv = layer(input_i, conv_weights=weights[i])
            output.append(conv)
        if self.unet:
            output = output[self.num_downsamples:]
        return output[::-1]


class WeightReshaper:
    """Handles all weight reshape related tasks."""

    def reshape_weight(self, x, weight_shape):
        """Reshape input x to the desired weight shape.

        Args:
            x (tensor or list of tensors): Input features.
            weight_shape (list of int): Desired shape of the weight.
        Returns:
            (tuple):
              - weight (tensor): Network weights
              - bias (tensor): Network bias.
        """
        if type(weight_shape[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_shape))
        if type(x) == list:
            return [self.reshape_weight(xi, wi) for xi, wi in zip(x, weight_shape)]
        weight_shape = [x.size(0)] + weight_shape
        bias_size = weight_shape[1]
        try:
            weight = x[:, :-bias_size].view(weight_shape)
            bias = x[:, -bias_size:]
        except Exception:
            weight = x.view(weight_shape)
            bias = None
        return [weight, bias]

    def split_weights(self, weight, sizes):
        """When the desired shape is a list, first divide the input to each
        corresponding weight shape in the list.

        Args:
            weight (tensor): Input weight.
            sizes (int or list of int): Target sizes.
        Returns:
            weight (list of tensors): Divided weights.
        """
        if isinstance(sizes, list):
            weights = []
            cur_size = 0
            for i in range(len(sizes)):
                next_size = cur_size + self.sum(sizes[i])
                weights.append(self.split_weights(weight[:, cur_size:next_size], sizes[i]))
                cur_size = next_size
            assert next_size == weight.size(1)
            return weights
        return weight

    def reshape_embed_input(self, x):
        """Reshape input to be (B x C) X H X W.

        Args:
            x (tensor or list of tensors): Input features.
        Returns:
            x (tensor or list of tensors): Reshaped features.
        """
        if isinstance(x, list):
            return [self.reshape_embed_input(xi) for xi in zip(x)]
        b, c, _, _ = x.size()
        x = x.view(b * c, -1)
        return x

    def sum(self, x):
        """Sum all elements recursively in a nested list.

        Args:
            x (nested list of int): Input list of elements.
        Returns:
            out (int): Sum of all elements.
        """
        if type(x) != list:
            return x
        return sum([self.sum(xi) for xi in x])

    def sum_mul(self, x):
        """Given a weight shape, compute the number of elements needed for
        weight + bias. If input is a list of shapes, sum all the elements.

        Args:
            x (list of int): Input list of elements.
        Returns:
            out (int or list of int): Summed number of elements.
        """
        assert type(x) == list
        if type(x[0]) != list:
            return np.prod(x) + x[0]
        return [self.sum_mul(xi) for xi in x]


def get_and_setattr(cfg, name, default):
    """Get attribute with default choice. If attribute does not exist, set it
    using the default value.

    Args:
        cfg (obj) : Config options.
        name (str) : Attribute name.
        default (obj) : Default attribute.

    Returns:
        (obj) : Desired attribute.
    """
    if not hasattr(cfg, name) or name not in cfg.__dict__:
        setattr(cfg, name, default)
    return getattr(cfg, name)


def get_nested_attr(cfg, attr_name, default):
    """Iteratively try to get the attribute from cfg. If not found, return
    default.

    Args:
        cfg (obj): Config file.
        attr_name (str): Attribute name (e.g. XXX.YYY.ZZZ).
        default (obj): Default return value for the attribute.

    Returns:
        (obj): Attribute value.
    """
    names = attr_name.split('.')
    atr = cfg
    for name in names:
        if not hasattr(atr, name):
            return default
        atr = getattr(atr, name)
    return atr


def get_paired_input_image_channel_number(data_cfg):
    """Get number of channels for the input image.

    Args:
        data_cfg (obj): Data configuration structure.
    Returns:
        num_channels (int): Number of input image channels.
    """
    num_channels = 0
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_image:
                num_channels += data_type[k].num_channels
                None
    None
    return num_channels


def get_paired_input_label_channel_number(data_cfg, video=False):
    """Get number of channels for the input label map.

    Args:
        data_cfg (obj): Data configuration structure.
        video (bool): Whether we are dealing with video data.
    Returns:
        num_channels (int): Number of input label map channels.
    """
    num_labels = 0
    if not hasattr(data_cfg, 'input_labels'):
        return num_labels
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_labels:
                if hasattr(data_cfg, 'one_hot_num_classes') and k in data_cfg.one_hot_num_classes:
                    num_labels += data_cfg.one_hot_num_classes[k]
                    if getattr(data_cfg, 'use_dont_care', False):
                        num_labels += 1
                else:
                    num_labels += data_type[k].num_channels
            None
    if video:
        num_time_steps = getattr(data_cfg.train, 'initial_sequence_length', None)
        num_labels *= num_time_steps
        num_labels += get_paired_input_image_channel_number(data_cfg) * (num_time_steps - 1)
    None
    return num_labels


class WeightGenerator(nn.Module):
    """Weight generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.embed_cfg = embed_cfg = gen_cfg.embed
        self.embed_arch = embed_cfg.arch
        num_filters = gen_cfg.num_filters
        self.max_num_filters = gen_cfg.max_num_filters
        self.num_downsamples = num_downsamples = gen_cfg.num_downsamples
        self.num_filters_each_layer = num_filters_each_layer = [min(self.max_num_filters, num_filters * 2 ** i) for i in range(num_downsamples + 2)]
        if getattr(embed_cfg, 'num_filters', 32) != num_filters:
            raise ValueError('Embedding network must have the same number of filters as generator.')
        hyper_cfg = gen_cfg.hyper
        kernel_size = getattr(hyper_cfg, 'kernel_size', 3)
        activation_norm_type = getattr(hyper_cfg, 'activation_norm_type', 'sync_batch')
        weight_norm_type = getattr(hyper_cfg, 'weight_norm_type', 'spectral')
        self.conv_kernel_size = conv_kernel_size = gen_cfg.kernel_size
        self.embed_kernel_size = embed_kernel_size = getattr(gen_cfg.embed, 'kernel_size', 3)
        self.kernel_size = kernel_size = getattr(gen_cfg.activation_norm_params, 'kernel_size', 1)
        self.spade_in_channels = []
        for i in range(num_downsamples + 1):
            self.spade_in_channels += [num_filters_each_layer[i]]
        self.use_hyper_spade = hyper_cfg.is_hyper_spade
        self.use_hyper_embed = hyper_cfg.is_hyper_embed
        self.use_hyper_conv = hyper_cfg.is_hyper_conv
        self.num_hyper_layers = hyper_cfg.num_hyper_layers
        order = getattr(gen_cfg.hyper, 'hyper_block_order', 'NAC')
        self.conv_before_norm = order.find('C') < order.find('N')
        self.concat_ref_label = 'concat' in hyper_cfg.method_to_use_ref_labels
        self.mul_ref_label = 'mul' in hyper_cfg.method_to_use_ref_labels
        self.sh_fix = self.sw_fix = 32
        self.num_fc_layers = getattr(hyper_cfg, 'num_fc_layers', 2)
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        elif get_nested_attr(data_cfg, 'for_pose_dataset.pose_type', 'both') == 'open':
            num_input_channels -= 3
        data_cfg.num_input_channels = num_input_channels
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_ref_channels = num_img_channels + (num_input_channels if self.concat_ref_label else 0)
        conv_2d_block = partial(Conv2dBlock, kernel_size=kernel_size, padding=kernel_size // 2, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='leakyrelu')
        self.ref_img_first = conv_2d_block(num_ref_channels, num_filters)
        if self.mul_ref_label:
            self.ref_label_first = conv_2d_block(num_input_channels, num_filters)
        for i in range(num_downsamples):
            in_ch, out_ch = num_filters_each_layer[i], num_filters_each_layer[i + 1]
            setattr(self, 'ref_img_down_%d' % i, conv_2d_block(in_ch, out_ch, stride=2))
            setattr(self, 'ref_img_up_%d' % i, conv_2d_block(out_ch, in_ch))
            if self.mul_ref_label:
                setattr(self, 'ref_label_down_%d' % i, conv_2d_block(in_ch, out_ch, stride=2))
                setattr(self, 'ref_label_up_%d' % i, conv_2d_block(out_ch, in_ch))
        if self.use_hyper_spade or self.use_hyper_conv:
            for i in range(self.num_hyper_layers):
                ch_in, ch_out = num_filters_each_layer[i], num_filters_each_layer[i + 1]
                conv_ks2 = conv_kernel_size ** 2
                embed_ks2 = embed_kernel_size ** 2
                spade_ks2 = kernel_size ** 2
                spade_in_ch = self.spade_in_channels[i]
                fc_names, fc_ins, fc_outs = [], [], []
                if self.use_hyper_spade:
                    fc0_out = fcs_out = (spade_in_ch * spade_ks2 + 1) * (1 if self.conv_before_norm else 2)
                    fc1_out = (spade_in_ch * spade_ks2 + 1) * (1 if ch_in != ch_out else 2)
                    fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
                    fc_ins += [ch_out] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                    if self.use_hyper_embed:
                        fc_names += ['fc_spade_e']
                        fc_ins += [ch_out]
                        fc_outs += [ch_in * embed_ks2 + 1]
                if self.use_hyper_conv:
                    fc0_out = ch_out * conv_ks2 + 1
                    fc1_out = ch_in * conv_ks2 + 1
                    fcs_out = ch_out + 1
                    fc_names += ['fc_conv_0', 'fc_conv_1', 'fc_conv_s']
                    fc_ins += [ch_in] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                linear_block = partial(LinearBlock, weight_norm_type='spectral', nonlinearity='leakyrelu')
                for n, l in enumerate(fc_names):
                    fc_in = fc_ins[n] if self.mul_ref_label else self.sh_fix * self.sw_fix
                    fc_layer = [linear_block(fc_in, ch_out)]
                    for k in range(1, self.num_fc_layers):
                        fc_layer += [linear_block(ch_out, ch_out)]
                    fc_layer += [LinearBlock(ch_out, fc_outs[n], weight_norm_type='spectral')]
                    setattr(self, '%s_%d' % (l, i), nn.Sequential(*fc_layer))
        num_hyper_layers = self.num_hyper_layers if self.use_hyper_embed else 0
        self.label_embedding = LabelEmbedder(self.embed_cfg, num_input_channels, num_hyper_layers=num_hyper_layers)
        if hasattr(hyper_cfg, 'attention'):
            self.num_downsample_atn = get_and_setattr(hyper_cfg.attention, 'num_downsamples', 2)
            if data_cfg.initial_few_shot_K > 1:
                self.attention_module = AttentionModule(hyper_cfg, data_cfg, conv_2d_block, num_filters_each_layer)
        else:
            self.num_downsample_atn = 0

    def forward(self, ref_image, ref_label, label, is_first_frame):
        """Generate network weights based on the reference images.

        Args:
            ref_image (NxKx3xHxW tensor): Reference images.
            ref_label (NxKxCxHxW tensor): Reference labels.
            label (NxCxHxW tensor): Target label.
            is_first_frame (bool): Whether the current frame is the first frame.

        Returns:
            (tuple):
              - x (NxC2xH2xW2 tensor): Encoded features from reference images
                for the main branch (as input to the decoder).
              - encoded_label (list of tensors): Encoded target label map for
                SPADE.
              - conv_weights (list of tensors): Network weights for conv
                layers in the main network.
              - norm_weights (list of tensors): Network weights for SPADE
                layers in the main network.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention
                scores.
              - ref_idx (Nx1 tensor): Index for which image to use from the
                reference images.
        """
        b, k, c, h, w = ref_image.size()
        ref_image = ref_image.view(b * k, -1, h, w)
        if ref_label is not None:
            ref_label = ref_label.view(b * k, -1, h, w)
        x, encoded_ref, atn, atn_vis, ref_idx = self.encode_reference(ref_image, ref_label, label, k)
        if self.training or is_first_frame or k > 1:
            embedding_weights, norm_weights, conv_weights = [], [], []
            for i in range(self.num_hyper_layers):
                if self.use_hyper_spade:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i + 1)]
                    embedding_weight, norm_weight = self.get_norm_weights(feat, i)
                    embedding_weights.append(embedding_weight)
                    norm_weights.append(norm_weight)
                if self.use_hyper_conv:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i)]
                    conv_weights.append(self.get_conv_weights(feat, i))
            if not self.training:
                self.embedding_weights, self.conv_weights, self.norm_weights = embedding_weights, conv_weights, norm_weights
        else:
            embedding_weights, conv_weights, norm_weights = self.embedding_weights, self.conv_weights, self.norm_weights
        encoded_label = self.label_embedding(label, weights=embedding_weights if self.use_hyper_embed else None)
        return x, encoded_label, conv_weights, norm_weights, atn, atn_vis, ref_idx

    def encode_reference(self, ref_image, ref_label, label, k):
        """Encode the reference image to get features for weight generation.

        Args:
            ref_image ((NxK)x3xHxW tensor): Reference images.
            ref_label ((NxK)xCxHxW tensor): Reference labels.
            label (NxCxHxW tensor): Target label.
            k (int): Number of reference images.
        Returns:
            (tuple):
              - x (NxC2xH2xW2 tensor): Encoded features from reference images
                for the main branch (as input to the decoder).
              - encoded_ref (list of tensors): Encoded features from reference
                images for the weight generation branch.
              - attention (Nx(KxH1xW1)x(H1xW1) tensor): Attention maps.
              - atn_vis (1x1xH1xW1 tensor): Visualization for attention scores.
              - ref_idx (Nx1 tensor): Index for which image to use from the
                reference images.
        """
        if self.concat_ref_label:
            concat_ref = torch.cat([ref_image, ref_label], dim=1)
            x = self.ref_img_first(concat_ref)
        elif self.mul_ref_label:
            x = self.ref_img_first(ref_image)
            x_label = self.ref_label_first(ref_label)
        else:
            x = self.ref_img_first(ref_image)
        atn = atn_vis = ref_idx = None
        for i in range(self.num_downsamples):
            x = getattr(self, 'ref_img_down_' + str(i))(x)
            if self.mul_ref_label:
                x_label = getattr(self, 'ref_label_down_' + str(i))(x_label)
            if k > 1 and i == self.num_downsample_atn - 1:
                x, atn, atn_vis = self.attention_module(x, label, ref_label)
                if self.mul_ref_label:
                    x_label, _, _ = self.attention_module(x_label, None, None, atn)
                atn_sum = atn.view(label.shape[0], k, -1).sum(2)
                ref_idx = torch.argmax(atn_sum, dim=1)
        encoded_image_ref = [x]
        if self.mul_ref_label:
            encoded_ref_label = [x_label]
        for i in reversed(range(self.num_downsamples)):
            conv = getattr(self, 'ref_img_up_' + str(i))(encoded_image_ref[-1])
            encoded_image_ref.append(conv)
            if self.mul_ref_label:
                conv_label = getattr(self, 'ref_label_up_' + str(i))(encoded_ref_label[-1])
                encoded_ref_label.append(conv_label)
        if self.mul_ref_label:
            encoded_ref = []
            for i in range(len(encoded_image_ref)):
                conv, conv_label = encoded_image_ref[i], encoded_ref_label[i]
                b, c, h, w = conv.size()
                conv_label = nn.Softmax(dim=1)(conv_label)
                conv_prod = (conv.view(b, c, 1, h * w) * conv_label.view(b, 1, c, h * w)).sum(3, keepdim=True)
                encoded_ref.append(conv_prod)
        else:
            encoded_ref = encoded_image_ref
        encoded_ref = encoded_ref[::-1]
        return x, encoded_ref, atn, atn_vis, ref_idx

    def get_norm_weights(self, x, i):
        """Adaptively generate weights for SPADE in layer i of generator.

        Args:
            x (NxCxHxW tensor): Input features.
            i (int): Layer index.
        Returns:
            (tuple):
              - embedding_weights (list of tensors): Weights for the label
                embedding network.
              - norm_weights (list of tensors): Weights for the SPADE layers.
        """
        if not self.mul_ref_label:
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        spade_ch = self.spade_in_channels[i]
        eks, sks = self.embed_kernel_size, self.kernel_size
        b = x.size(0)
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)
        embedding_weights = None
        if self.use_hyper_embed:
            fc_e = getattr(self, 'fc_spade_e_' + str(i))(x).view(b, -1)
            if 'decoder' in self.embed_arch:
                weight_shape = [in_ch, out_ch, eks, eks]
                fc_e = fc_e[:, :-in_ch]
            else:
                weight_shape = [out_ch, in_ch, eks, eks]
            embedding_weights = weight_reshaper.reshape_weight(fc_e, weight_shape)
        fc_0 = getattr(self, 'fc_spade_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_spade_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_spade_s_' + str(i))(x).view(b, -1)
        if self.conv_before_norm:
            out_ch = in_ch
        weight_0 = weight_reshaper.reshape_weight(fc_0, [out_ch * 2, spade_ch, sks, sks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch * 2, spade_ch, sks, sks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [out_ch * 2, spade_ch, sks, sks])
        norm_weights = [weight_0, weight_1, weight_s]
        return embedding_weights, norm_weights

    def get_conv_weights(self, x, i):
        """Adaptively generate weights for layer i in main branch convolutions.

        Args:
            x (NxCxHxW tensor): Input features.
            i (int): Layer index.
        Returns:
            (tuple):
              - conv_weights (list of tensors): Weights for the conv layers in
                the main branch.
        """
        if not self.mul_ref_label:
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        cks = self.conv_kernel_size
        b = x.size()[0]
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)
        fc_0 = getattr(self, 'fc_conv_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_conv_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_conv_s_' + str(i))(x).view(b, -1)
        weight_0 = weight_reshaper.reshape_weight(fc_0, [in_ch, out_ch, cks, cks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch, in_ch, cks, cks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [in_ch, out_ch, 1, 1])
        return [weight_0, weight_1, weight_s]

    def reset(self):
        """Reset the network at the beginning of a sequence."""
        self.embedding_weights = self.conv_weights = self.norm_weights = None


class BaseNetwork(nn.Module):
    """vid2vid generator."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_num_filters(self, num_downsamples):
        """Get the number of filters at current layer.

        Args:
            num_downsamples (int) : How many downsamples at current layer.
        Returns:
            output (int) : Number of filters.
        """
        return min(self.max_num_filters, self.num_filters * 2 ** num_downsamples)


class FlowGenerator(BaseNetwork):
    """Flow generator constructor.

    Args:
       flow_cfg (obj): Flow definition part of the yaml config file.
       data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, flow_cfg, data_cfg):
        super().__init__()
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        num_prev_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_frames = data_cfg.num_frames_G
        self.num_filters = num_filters = getattr(flow_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(flow_cfg, 'max_num_filters', 1024)
        num_downsamples = getattr(flow_cfg, 'num_downsamples', 5)
        kernel_size = getattr(flow_cfg, 'kernel_size', 3)
        padding = kernel_size // 2
        self.num_res_blocks = getattr(flow_cfg, 'num_res_blocks', 6)
        self.flow_output_multiplier = getattr(flow_cfg, 'flow_output_multiplier', 20)
        activation_norm_type = getattr(flow_cfg, 'activation_norm_type', 'sync_batch')
        weight_norm_type = getattr(flow_cfg, 'weight_norm_type', 'spectral')
        base_conv_block = partial(Conv2dBlock, kernel_size=kernel_size, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='leakyrelu')
        down_lbl = [base_conv_block(num_input_channels * num_frames, num_filters)]
        down_img = [base_conv_block(num_prev_img_channels * (num_frames - 1), num_filters)]
        for i in range(num_downsamples):
            down_lbl += [base_conv_block(self.get_num_filters(i), self.get_num_filters(i + 1), stride=2)]
            down_img += [base_conv_block(self.get_num_filters(i), self.get_num_filters(i + 1), stride=2)]
        res_flow = []
        ch = self.get_num_filters(num_downsamples)
        for i in range(self.num_res_blocks):
            res_flow += [Res2dBlock(ch, ch, kernel_size, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, order='CNACN')]
        up_flow = []
        for i in reversed(range(num_downsamples)):
            up_flow += [nn.Upsample(scale_factor=2), base_conv_block(self.get_num_filters(i + 1), self.get_num_filters(i))]
        conv_flow = [Conv2dBlock(num_filters, 2, kernel_size, padding=padding)]
        conv_mask = [Conv2dBlock(num_filters, 1, kernel_size, padding=padding, nonlinearity='sigmoid')]
        self.down_lbl = nn.Sequential(*down_lbl)
        self.down_img = nn.Sequential(*down_img)
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)
        self.conv_mask = nn.Sequential(*conv_mask)

    def forward(self, label, img_prev):
        """Flow generator forward.

        Args:
           label (4D tensor) : Input label tensor.
           img_prev (4D tensor) : Previously generated image tensors.
        Returns:
            (tuple):
              - flow (4D tensor) : Generated flow map.
              - mask (4D tensor) : Generated occlusion mask.
        """
        downsample = self.down_lbl(label) + self.down_img(img_prev)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_output_multiplier
        mask = self.conv_mask(flow_feat)
        return flow, mask


class FUNITTranslator(nn.Module):
    """

    Args:
         num_filters (int): Base filter numbers.
         num_filters_mlp (int): Base filter number in the MLP module.
         style_dims (int): Dimension of the style code.
         num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
         num_mlp_blocks (int): Number of layers in the MLP module.
         num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
         num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
         num_image_channels (int): Number of input image channels.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
    """

    def __init__(self, num_filters=64, num_filters_mlp=256, style_dims=64, num_res_blocks=2, num_mlp_blocks=3, num_downsamples_style=4, num_downsamples_content=2, num_image_channels=3, weight_norm_type='', **kwargs):
        super().__init__()
        self.style_encoder = StyleEncoder(num_downsamples_style, num_image_channels, num_filters, style_dims, 'reflect', 'none', weight_norm_type, 'relu')
        self.content_encoder = ContentEncoder(num_downsamples_content, num_res_blocks, num_image_channels, num_filters, 'reflect', 'instance', weight_norm_type, 'relu')
        self.decoder = Decoder(self.content_encoder.output_dim, num_filters_mlp, num_image_channels, num_downsamples_content, 'reflect', weight_norm_type, 'relu')
        self.mlp = MLP(style_dims, num_filters_mlp, num_filters_mlp, num_mlp_blocks, 'none', 'relu')

    def forward(self, images):
        """Reconstruct the input image by combining the computer content and
        style code.

        Args:
            images (tensor): Input image tensor.
        """
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        """Encoder images to get their content and style codes.

        Args:
            images (tensor): Input image tensor.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        """Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        style = self.mlp(style)
        images = self.decoder(content, style)
        return images


class AffineMod(nn.Module):
    """Learning affine modulation of activation.

    Args:
        in_features (int): Number of input features.
        style_features (int): Number of style features.
        mod_bias (bool): Whether to modulate bias.
    """

    def __init__(self, in_features, style_features, mod_bias=True):
        super().__init__()
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        if mod_bias:
            self.weight_beta = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([in_features], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])
        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)
        x = x * alpha
        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)
            x = x + beta
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class ModLinear(nn.Module):
    """Linear layer with affine modulation (Based on StyleGAN2 mod demod).
    Equivalent to affine modulation following linear, but faster when the same modulation parameters are shared across
    multiple inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        style_features (int): Number of style features.
        bias (bool): Apply additive bias before the activation function?
        mod_bias (bool): Whether to modulate bias.
        output_mode (bool): If True, modulate output instead of input.
        weight_gain (float): Initialization gain
    """

    def __init__(self, in_features, out_features, style_features, bias=True, mod_bias=True, output_mode=False, weight_gain=1, bias_init=0):
        super().__init__()
        weight_gain = weight_gain / np.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) * weight_gain)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        self.output_mode = output_mode
        if mod_bias:
            if output_mode:
                mod_bias_dims = out_features
            else:
                mod_bias_dims = in_features
            self.weight_beta = nn.Parameter(torch.randn([mod_bias_dims, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([mod_bias_dims], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])
        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)
        w = self.weight
        w = w.unsqueeze(0) * alpha
        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)
            if not self.output_mode:
                x = x + beta
        b = self.bias
        if b is not None:
            b = b[None, None, :]
        if self.mod_bias and self.output_mode:
            if b is None:
                b = beta
            else:
                b = b + beta
        if b is not None:
            x = torch.baddbmm(b, x, w.transpose(1, 2))
        else:
            x = x.bmm(w.transpose(1, 2))
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class RenderMLP(nn.Module):
    """ MLP with affine modulation."""

    def __init__(self, in_channels, style_dim, viewdir_dim, mask_dim=680, out_channels_s=1, out_channels_c=3, hidden_channels=256, use_seg=True):
        super(RenderMLP, self).__init__()
        self.use_seg = use_seg
        if self.use_seg:
            self.fc_m_a = nn.Linear(mask_dim, hidden_channels, bias=False)
        self.fc_viewdir = None
        if viewdir_dim > 0:
            self.fc_viewdir = nn.Linear(viewdir_dim, hidden_channels, bias=False)
        self.fc_1 = nn.Linear(in_channels, hidden_channels)
        self.fc_2 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_3 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_4 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_sigma = nn.Linear(hidden_channels, out_channels_s)
        if viewdir_dim > 0:
            self.fc_5 = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.mod_5 = AffineMod(hidden_channels, style_dim, mod_bias=True)
        else:
            self.fc_5 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_6 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, raydir, z, m):
        """ Forward network

        Args:
            x (N x H x W x M x in_channels tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            z (N x style_dim tensor): Style codes.
            m (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        b, h, w, n, _ = x.size()
        z = z[:, None, None, None, :]
        f = self.fc_1(x)
        if self.use_seg:
            f = f + self.fc_m_a(m)
        f = self.act(f)
        f = self.act(self.fc_2(f, z))
        f = self.act(self.fc_3(f, z))
        f = self.act(self.fc_4(f, z))
        sigma = self.fc_sigma(f)
        if self.fc_viewdir is not None:
            f = self.fc_5(f)
            f = f + self.fc_viewdir(raydir)
            f = self.act(self.mod_5(f, z))
        else:
            f = self.act(self.fc_5(f, z))
        f = self.act(self.fc_6(f, z))
        c = self.fc_out_c(f)
        return sigma, c


class StyleMLP(nn.Module):
    """MLP converting style code to intermediate style representation."""

    def __init__(self, style_dim, out_dim, hidden_channels=256, leaky_relu=True, num_layers=5, normalize_input=True, output_act=True):
        super(StyleMLP, self).__init__()
        self.normalize_input = normalize_input
        self.output_act = output_act
        fc_layers = []
        fc_layers.append(nn.Linear(style_dim, hidden_channels, bias=True))
        for i in range(num_layers - 1):
            fc_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_out = nn.Linear(hidden_channels, out_dim, bias=True)
        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def forward(self, z):
        """ Forward network

        Args:
            z (N x style_dim tensor): Style codes.
        """
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=-1)
        for fc_layer in self.fc_layers:
            z = self.act(fc_layer(z))
        z = self.fc_out(z)
        if self.output_act:
            z = self.act(z)
        return z


class SKYMLP(nn.Module):
    """MLP converting ray directions to sky features."""

    def __init__(self, in_channels, style_dim, out_channels_c=3, hidden_channels=256, leaky_relu=True):
        super(SKYMLP, self).__init__()
        self.fc_z_a = nn.Linear(style_dim, hidden_channels, bias=False)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)
        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def forward(self, x, z):
        """Forward network

        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """
        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1)
        y = self.act(self.fc1(x) + z)
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))
        c = self.fc_out_c(y)
        return c


class RenderCNN(nn.Module):
    """CNN converting intermediate feature map to final image."""

    def __init__(self, in_channels, style_dim, hidden_channels=256, leaky_relu=True):
        super(RenderCNN, self).__init__()
        self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv2a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0)
        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = functools.partial(F.relu, inplace=True)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w + 1) + b

    def forward(self, x, z):
        """Forward network.

        Args:
            x (N x in_channels x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        z = self.fc_z_cond(z)
        adapt = torch.chunk(z, 2 * 2, dim=-1)
        y = self.act(self.conv1(x))
        y = y + self.conv2b(self.act(self.conv2a(y)))
        y = self.act(self.modulate(y, adapt[0], adapt[1]))
        y = y + self.conv3b(self.act(self.conv3a(y)))
        y = self.act(self.modulate(y, adapt[2], adapt[3]))
        y = y + self.conv4b(self.act(self.conv4a(y)))
        y = self.act(y)
        y = self.conv4(y)
        return y


class Base3DGenerator(nn.Module):
    """Minecraft 3D generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Base3DGenerator, self).__init__()
        None
        self.pe_no_pe_feat_dim = getattr(gen_cfg, 'pe_no_pe_feat_dim', 0)
        input_dim = (gen_cfg.blk_feat_dim - self.pe_no_pe_feat_dim) * (gen_cfg.pe_lvl_feat * 2) + self.pe_no_pe_feat_dim
        if gen_cfg.pe_incl_orig_feat:
            input_dim += gen_cfg.blk_feat_dim - self.pe_no_pe_feat_dim
        None
        self.input_dim = input_dim
        self.mlp_model_kwargs = gen_cfg.mlp_model_kwargs
        self.pe_lvl_localcoords = getattr(gen_cfg, 'pe_lvl_localcoords', 0)
        if self.pe_lvl_localcoords > 0:
            self.mlp_model_kwargs['poscode_dim'] = self.pe_lvl_localcoords * 2 * 3
        input_dim_viewdir = 3 * (gen_cfg.pe_lvl_raydir * 2)
        if gen_cfg.pe_incl_orig_raydir:
            input_dim_viewdir += 3
        None
        self.input_dim_viewdir = input_dim_viewdir
        self.pe_params = [gen_cfg.pe_lvl_feat, gen_cfg.pe_incl_orig_feat, gen_cfg.pe_lvl_raydir, gen_cfg.pe_incl_orig_raydir]
        style_dims = gen_cfg.style_dims
        self.style_dims = style_dims
        interm_style_dims = getattr(gen_cfg, 'interm_style_dims', style_dims)
        self.interm_style_dims = interm_style_dims
        self.style_net = globals()[gen_cfg.stylenet_model](style_dims, interm_style_dims, **gen_cfg.stylenet_model_kwargs)
        final_feat_dim = getattr(gen_cfg, 'final_feat_dim', 16)
        self.final_feat_dim = final_feat_dim
        sky_input_dim_base = 3
        sky_input_dim = sky_input_dim_base * (gen_cfg.pe_lvl_raydir_sky * 2)
        if gen_cfg.pe_incl_orig_raydir_sky:
            sky_input_dim += sky_input_dim_base
        None
        self.pe_params_sky = [gen_cfg.pe_lvl_raydir_sky, gen_cfg.pe_incl_orig_raydir_sky]
        self.sky_net = SKYMLP(sky_input_dim, style_dim=interm_style_dims, out_channels_c=final_feat_dim)
        style_enc_cfg = getattr(gen_cfg, 'style_enc', None)
        setattr(style_enc_cfg, 'input_image_channels', 3)
        setattr(style_enc_cfg, 'style_dims', gen_cfg.style_dims)
        self.style_encoder = StyleEncoder(style_enc_cfg)
        self.num_blocks_early_stop = gen_cfg.num_blocks_early_stop
        self.num_samples = gen_cfg.num_samples
        self.sample_depth = gen_cfg.sample_depth
        self.coarse_deterministic_sampling = getattr(gen_cfg, 'coarse_deterministic_sampling', True)
        self.sample_use_box_boundaries = getattr(gen_cfg, 'sample_use_box_boundaries', True)
        self.raw_noise_std = getattr(gen_cfg, 'raw_noise_std', 0.0)
        self.dists_scale = getattr(gen_cfg, 'dists_scale', 0.25)
        self.clip_feat_map = getattr(gen_cfg, 'clip_feat_map', True)
        self.keep_sky_out = getattr(gen_cfg, 'keep_sky_out', False)
        self.keep_sky_out_avgpool = getattr(gen_cfg, 'keep_sky_out_avgpool', False)
        keep_sky_out_learnbg = getattr(gen_cfg, 'keep_sky_out_learnbg', False)
        self.sky_global_avgpool = getattr(gen_cfg, 'sky_global_avgpool', False)
        if self.keep_sky_out:
            self.sky_replace_color = None
            if keep_sky_out_learnbg:
                sky_replace_color = torch.zeros([final_feat_dim])
                sky_replace_color.requires_grad = True
                self.sky_replace_color = torch.nn.Parameter(sky_replace_color)
        self.denoiser = RenderCNN(final_feat_dim, style_dim=interm_style_dims)
        self.pad = gen_cfg.pad

    def get_param_groups(self, cfg_opt):
        None
        if hasattr(cfg_opt, 'ignore_parameters'):
            None
            optimize_parameters = []
            for k, x in self.named_parameters():
                match = False
                for m in cfg_opt.ignore_parameters:
                    if re.match(m, k) is not None:
                        match = True
                        None
                        break
                if match is False:
                    None
                    optimize_parameters.append(x)
        else:
            optimize_parameters = self.parameters()
        param_groups = []
        param_groups.append({'params': optimize_parameters})
        if hasattr(cfg_opt, 'param_groups'):
            optimized_param_names = []
            all_param_names = [k for k, v in self.named_parameters()]
            param_groups = []
            for k, v in cfg_opt.param_groups.items():
                None
                params = getattr(self, k)
                named_parameters = [k]
                if issubclass(type(params), nn.Module):
                    named_parameters = [(k + '.' + pname) for pname, _ in params.named_parameters()]
                    params = params.parameters()
                param_groups.append({'params': params, **v})
                optimized_param_names.extend(named_parameters)
        None
        return param_groups

    def _forward_perpix_sub(self, blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot=None):
        """Forwarding the MLP.

        Args:
            blk_feats (K x C1 tensor): Sparse block features.
            worldcoord2 (N x H x W x L x 3 tensor): 3D world coordinates of sampled points.
            raydirs_in (N x H x W x 1 x C2 tensor or None): ray direction embeddings.
            z (N x C3 tensor): Intermediate style vectors.
            mc_masks_onehot (N x H x W x L x C4): One-hot segmentation maps.
        Returns:
            net_out_s (N x H x W x L x 1 tensor): Opacities.
            net_out_c (N x H x W x L x C5 tensor): Color embeddings.
        """
        proj_feature = voxlib.sparse_trilinear_interp_worldcoord(blk_feats, self.voxel.corner_t, worldcoord2, ign_zero=True)
        render_net_extra_kwargs = {}
        if self.pe_lvl_localcoords > 0:
            local_coords = torch.remainder(worldcoord2, 1.0) * 2.0
            local_coords[torch.isnan(local_coords)] = 0.0
            local_coords = local_coords.contiguous()
            poscode = voxlib.positional_encoding(local_coords, self.pe_lvl_localcoords, -1, False)
            render_net_extra_kwargs['poscode'] = poscode
        if self.pe_params[0] == 0 and self.pe_params[1] is True:
            feature_in = proj_feature
        elif self.pe_no_pe_feat_dim > 0:
            feature_in = voxlib.positional_encoding(proj_feature[..., :-self.pe_no_pe_feat_dim].contiguous(), self.pe_params[0], -1, self.pe_params[1])
            feature_in = torch.cat([feature_in, proj_feature[..., -self.pe_no_pe_feat_dim:]], dim=-1)
        else:
            feature_in = voxlib.positional_encoding(proj_feature.contiguous(), self.pe_params[0], -1, self.pe_params[1])
        net_out_s, net_out_c = self.render_net(feature_in, raydirs_in, z, mc_masks_onehot, **render_net_extra_kwargs)
        if self.raw_noise_std > 0.0:
            noise = torch.randn_like(net_out_s) * self.raw_noise_std
            net_out_s = net_out_s + noise
        return net_out_s, net_out_c

    def _forward_perpix(self, blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z):
        """Sample points along rays, forwarding the per-point MLP and aggregate pixel features

        Args:
            blk_feats (K x C1 tensor): Sparse block features.
            voxel_id (N x H x W x M x 1 tensor): Voxel ids from ray-voxel intersection test. M: num intersected voxels
            depth2 (N x 2 x H x W x M x 1 tensor): Depths of entrance and exit points for each ray-voxel intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
            z (N x C3 tensor): Intermediate style vectors.
        """
        with torch.no_grad():
            raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
            if self.pe_params[2] == 0 and self.pe_params[3] is True:
                raydirs_in = raydirs_in
            elif self.pe_params[2] == 0 and self.pe_params[3] is False:
                raydirs_in = None
            else:
                raydirs_in = voxlib.positional_encoding(raydirs_in, self.pe_params[2], -1, self.pe_params[3])
            sky_mask = voxel_id[:, :, :, [-1], :] == 0
            sky_only_mask = voxel_id[:, :, :, [0], :] == 0
        with torch.no_grad():
            num_samples = self.num_samples + 1
            if self.sample_use_box_boundaries:
                num_samples = self.num_samples - self.num_blocks_early_stop
            rand_depth, new_dists, new_idx = mc_utils.sample_depth_batched(depth2, num_samples, deterministic=self.coarse_deterministic_sampling, use_box_boundaries=self.sample_use_box_boundaries, sample_depth=self.sample_depth)
            worldcoord2 = raydirs * rand_depth + cam_ori_t[:, None, None, None, :]
            voxel_id_reduced = self.label_trans.mc2reduced(voxel_id, ign2dirt=True)
            mc_masks = torch.gather(voxel_id_reduced, -2, new_idx)
            mc_masks = mc_masks.long()
            mc_masks_onehot = torch.zeros([mc_masks.size(0), mc_masks.size(1), mc_masks.size(2), mc_masks.size(3), self.num_reduced_labels], dtype=torch.float, device=voxel_id.device)
            mc_masks_onehot.scatter_(-1, mc_masks, 1.0)
        net_out_s, net_out_c = self._forward_perpix_sub(blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot)
        sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
        sky_raydirs_in = voxlib.positional_encoding(sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
        skynet_out_c = self.sky_net(sky_raydirs_in, z)
        weights = mc_utils.volum_rendering_relu(net_out_s, new_dists * self.dists_scale, dim=-2)
        weights = weights * torch.logical_not(sky_only_mask).float()
        total_weights_raw = torch.sum(weights, dim=-2, keepdim=True)
        total_weights = total_weights_raw
        is_gnd = worldcoord2[..., [0]] <= 1.0
        is_gnd = is_gnd.any(dim=-2, keepdim=True)
        nosky_mask = torch.logical_or(torch.logical_not(sky_mask), is_gnd)
        nosky_mask = nosky_mask.float()
        sky_weight = 1.0 - total_weights
        if self.keep_sky_out:
            if self.sky_replace_color is None or self.keep_sky_out_avgpool:
                if self.keep_sky_out_avgpool:
                    if hasattr(self, 'sky_avg'):
                        sky_avg = self.sky_avg
                    elif self.sky_global_avgpool:
                        sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                    else:
                        skynet_out_c_nchw = skynet_out_c.permute(0, 4, 1, 2, 3).squeeze(-1)
                        sky_avg = F.avg_pool2d(skynet_out_c_nchw, 31, stride=1, padding=15, count_include_pad=False)
                        sky_avg = sky_avg.permute(0, 2, 3, 1).unsqueeze(-2)
                    skynet_out_c = skynet_out_c * (1.0 - nosky_mask) + sky_avg * nosky_mask
                else:
                    sky_weight = sky_weight * (1.0 - nosky_mask)
            else:
                skynet_out_c = skynet_out_c * (1.0 - nosky_mask) + self.sky_replace_color * nosky_mask
        if self.clip_feat_map is True:
            rgbs = torch.clamp(net_out_c, -1, 1) + 1
            rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
            net_out = torch.sum(weights * rgbs, dim=-2, keepdim=True) + sky_weight * rgbs_sky
            net_out = net_out.squeeze(-2)
            net_out = net_out - 1
        elif self.clip_feat_map is False:
            rgbs = net_out_c
            rgbs_sky = skynet_out_c
            net_out = torch.sum(weights * rgbs, dim=-2, keepdim=True) + sky_weight * rgbs_sky
            net_out = net_out.squeeze(-2)
        elif self.clip_feat_map == 'tanh':
            rgbs = torch.tanh(net_out_c)
            rgbs_sky = torch.tanh(skynet_out_c)
            net_out = torch.sum(weights * rgbs, dim=-2, keepdim=True) + sky_weight * rgbs_sky
            net_out = net_out.squeeze(-2)
        else:
            raise NotImplementedError
        return net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, nosky_mask, sky_mask, sky_only_mask, new_idx

    def _forward_global(self, net_out, z):
        """Forward the CNN

        Args:
            net_out (N x C5 x H x W tensor): Intermediate feature maps.
            z (N x C3 tensor): Intermediate style vectors.

        Returns:
            fake_images (N x 3 x H x W tensor): Output image.
            fake_images_raw (N x 3 x H x W tensor): Output image before TanH.
        """
        fake_images = net_out.permute(0, 3, 1, 2)
        fake_images_raw = self.denoiser(fake_images, z)
        fake_images = torch.tanh(fake_images_raw)
        return fake_images, fake_images_raw


class AutoEncoder(nn.Module):
    """Improved UNIT autoencoder.

    Args:
        num_filters (int): Base filter numbers.
        max_num_filters (int): Maximum number of filters in the encoder.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_image_channels (int): Number of input image channels.
        content_norm_type (str): Type of activation normalization in the
            content encoder.
        decoder_norm_type (str): Type of activation normalization in the
            decoder.
        weight_norm_type (str): Type of weight normalization.
        output_nonlinearity (str): Type of nonlinearity before final output,
            ``'tanh'`` or ``'none'``.
        pre_act (bool): If ``True``, uses pre-activation residual blocks.
        apply_noise (bool): If ``True``, injects Gaussian noise in the decoder.
    """

    def __init__(self, num_filters=64, max_num_filters=256, num_res_blocks=4, num_downsamples_content=2, num_image_channels=3, content_norm_type='instance', decoder_norm_type='instance', weight_norm_type='', output_nonlinearity='', pre_act=False, apply_noise=False, **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn("Generator argument '{}' is not used.".format(key))
        self.content_encoder = ContentEncoder(num_downsamples_content, num_res_blocks, num_image_channels, num_filters, max_num_filters, 'reflect', content_norm_type, weight_norm_type, 'relu', pre_act)
        self.decoder = Decoder(num_downsamples_content, num_res_blocks, self.content_encoder.output_dim, num_image_channels, 'reflect', decoder_norm_type, weight_norm_type, 'relu', output_nonlinearity, pre_act, apply_noise)

    def forward(self, images):
        """Reconstruct an image.

        Args:
            images (Tensor): Input images.
        Returns:
            images_recon (Tensor): Reconstructed images.
        """
        content = self.content_encoder(images)
        images_recon = self.decoder(content)
        return images_recon


class LocalEnhancer(nn.Module):
    """Local enhancer constructor. These are sub-networks that are useful
    when aiming to produce high-resolution outputs.

    Args:
        gen_cfg (obj): local generator definition part of the yaml config
        file.
        data_cfg (obj): Data definition part of the yaml config file.
        num_input_channels (int): Number of segmentation labels.
        num_filters (int): Number of filters for the first layer.
        padding_mode (str): zero | reflect | ...
        base_conv_block (obj): Conv block with preset attributes.
        base_res_block (obj): Residual block with preset attributes.
        output_img (bool): Output is image or feature map.
    """

    def __init__(self, gen_cfg, data_cfg, num_input_channels, num_filters, padding_mode, base_conv_block, base_res_block, output_img=False):
        super(LocalEnhancer, self).__init__()
        num_res_blocks = getattr(gen_cfg, 'num_res_blocks', 3)
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        model_downsample = [base_conv_block(num_input_channels, num_filters, 7, padding=3), base_conv_block(num_filters, num_filters * 2, 3, stride=2, padding=1)]
        model_upsample = []
        for i in range(num_res_blocks):
            model_upsample += [base_res_block(num_filters * 2, num_filters * 2, 3, padding=1)]
        model_upsample += [NearestUpsample(scale_factor=2), base_conv_block(num_filters * 2, num_filters, 3, padding=1)]
        if output_img:
            model_upsample += [Conv2dBlock(num_filters, num_img_channels, 7, padding=3, padding_mode=padding_mode, nonlinearity='tanh')]
        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_upsample = nn.Sequential(*model_upsample)

    def forward(self, output_coarse, input_fine):
        """Local enhancer forward.

        Args:
            output_coarse (4D tensor) : Coarse output from previous layer.
            input_fine (4D tensor) : Fine input from current layer.
        Returns:
            output (4D tensor) : Refined output.
        """
        output = self.model_upsample(self.model_downsample(input_fine) + output_coarse)
        return output


class GlobalGenerator(nn.Module):
    """Coarse generator constructor. This is the main generator in the
    pix2pixHD architecture.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        num_input_channels (int): Number of segmentation labels.
        padding_mode (str): zero | reflect | ...
        base_conv_block (obj): Conv block with preset attributes.
        base_res_block (obj): Residual block with preset attributes.
    """

    def __init__(self, gen_cfg, data_cfg, num_input_channels, padding_mode, base_conv_block, base_res_block):
        super(GlobalGenerator, self).__init__()
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_filters = getattr(gen_cfg, 'num_filters', 64)
        num_downsamples = getattr(gen_cfg, 'num_downsamples', 4)
        num_res_blocks = getattr(gen_cfg, 'num_res_blocks', 9)
        model = [base_conv_block(num_input_channels, num_filters, kernel_size=7, padding=3)]
        for i in range(num_downsamples):
            ch = num_filters * 2 ** i
            model += [base_conv_block(ch, ch * 2, 3, padding=1, stride=2)]
        ch = num_filters * 2 ** num_downsamples
        for i in range(num_res_blocks):
            model += [base_res_block(ch, ch, 3, padding=1)]
        num_upsamples = num_downsamples
        for i in reversed(range(num_upsamples)):
            ch = num_filters * 2 ** i
            model += [NearestUpsample(scale_factor=2), base_conv_block(ch * 2, ch, 3, padding=1)]
        model += [Conv2dBlock(num_filters, num_img_channels, 7, padding=3, padding_mode=padding_mode, nonlinearity='tanh')]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Coarse-to-fine generator forward.

        Args:
            input (4D tensor) : Input semantic representations.
        Returns:
            output (4D tensor) : Synthesized image by generator.
        """
        return self.model(input)


class Encoder(nn.Module):
    """Encoder for getting region-wise features for style control.

    Args:
        enc_cfg (obj): Encoder definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, enc_cfg, data_cfg):
        super(Encoder, self).__init__()
        label_nc = get_paired_input_label_channel_number(data_cfg)
        feat_nc = enc_cfg.num_feat_channels
        n_clusters = getattr(enc_cfg, 'num_clusters', 10)
        for i in range(label_nc):
            dummy_arr = np.zeros((n_clusters, feat_nc), dtype=np.float32)
            self.register_buffer('cluster_%d' % i, torch.tensor(dummy_arr, dtype=torch.float32))
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        self.num_feat_channels = getattr(enc_cfg, 'num_feat_channels', 3)
        num_filters = getattr(enc_cfg, 'num_filters', 64)
        num_downsamples = getattr(enc_cfg, 'num_downsamples', 4)
        weight_norm_type = getattr(enc_cfg, 'weight_norm_type', 'none')
        activation_norm_type = getattr(enc_cfg, 'activation_norm_type', 'instance')
        padding_mode = getattr(enc_cfg, 'padding_mode', 'reflect')
        base_conv_block = partial(Conv2dBlock, padding_mode=padding_mode, weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='relu')
        model = [base_conv_block(num_img_channels, num_filters, 7, padding=3)]
        for i in range(num_downsamples):
            ch = num_filters * 2 ** i
            model += [base_conv_block(ch, ch * 2, 3, stride=2, padding=1)]
        for i in reversed(range(num_downsamples)):
            ch = num_filters * 2 ** i
            model += [NearestUpsample(scale_factor=2), base_conv_block(ch * 2, ch, 3, padding=1)]
        model += [Conv2dBlock(num_filters, self.num_feat_channels, 7, padding=3, padding_mode=padding_mode, nonlinearity='tanh')]
        self.model = nn.Sequential(*model)

    def forward(self, input, instance_map):
        """Extracting region-wise features

        Args:
            input (4D tensor): Real RGB images.
            instance_map (4D tensor): Instance label mask.
        Returns:
            outputs_mean (4D tensor): Instance-wise average-pooled
                feature maps.
        """
        outputs = self.model(input)
        outputs_mean = torch.zeros_like(outputs)
        inst_list = np.unique(instance_map.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size(0)):
                indices = (instance_map[b:b + 1] == int(i)).nonzero()
                for j in range(self.num_feat_channels):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class SPADEGenerator(nn.Module):
    """SPADE Image Generator constructor.

    Args:
        num_labels (int): Number of different labels.
        out_image_small_side_size (int): min(width, height)
        image_channels (int): Num. of channels of the output image.
        num_filters (int): Base filter numbers.
        kernel_size (int): Convolution kernel size.
        style_dims (int): Dimensions of the style code.
        activation_norm_params (obj): Spatially adaptive normalization param.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        global_adaptive_norm_type (str): Type of normalization in SPADE.
        skip_activation_norm (bool): If ``True``, applies activation norm to the
            shortcut connection in residual blocks.
        use_style_encoder (bool): Whether to use global adaptive norm
            like conditional batch norm or adaptive instance norm.
        output_multiplier (float): A positive number multiplied to the output
    """

    def __init__(self, num_labels, out_image_small_side_size, image_channels, num_filters, kernel_size, style_dims, activation_norm_params, weight_norm_type, global_adaptive_norm_type, skip_activation_norm, use_posenc_in_input_layer, use_style_encoder, output_multiplier):
        super(SPADEGenerator, self).__init__()
        self.output_multiplier = output_multiplier
        self.use_style_encoder = use_style_encoder
        self.use_posenc_in_input_layer = use_posenc_in_input_layer
        self.out_image_small_side_size = out_image_small_side_size
        self.num_filters = num_filters
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        activation_norm_type = 'spatially_adaptive'
        base_res2d_block = functools.partial(Res2dBlock, kernel_size=kernel_size, padding=padding, bias=[True, True, False], weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params, skip_activation_norm=skip_activation_norm, nonlinearity=nonlinearity, order='NACNAC')
        if self.use_style_encoder:
            self.fc_0 = LinearBlock(style_dims, 2 * style_dims, weight_norm_type=weight_norm_type, nonlinearity='relu', order='CAN')
            self.fc_1 = LinearBlock(2 * style_dims, 2 * style_dims, weight_norm_type=weight_norm_type, nonlinearity='relu', order='CAN')
            adaptive_norm_params = types.SimpleNamespace()
            if not hasattr(adaptive_norm_params, 'cond_dims'):
                setattr(adaptive_norm_params, 'cond_dims', 2 * style_dims)
            if not hasattr(adaptive_norm_params, 'activation_norm_type'):
                setattr(adaptive_norm_params, 'activation_norm_type', global_adaptive_norm_type)
            if not hasattr(adaptive_norm_params, 'weight_norm_type'):
                setattr(adaptive_norm_params, 'weight_norm_type', activation_norm_params.weight_norm_type)
            if not hasattr(adaptive_norm_params, 'separate_projection'):
                setattr(adaptive_norm_params, 'separate_projection', activation_norm_params.separate_projection)
            adaptive_norm_params.activation_norm_params = types.SimpleNamespace()
            setattr(adaptive_norm_params.activation_norm_params, 'affine', activation_norm_params.activation_norm_params.affine)
            base_cbn2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, stride=1, padding=padding, bias=True, weight_norm_type=weight_norm_type, activation_norm_type='adaptive', activation_norm_params=adaptive_norm_params, nonlinearity=nonlinearity, order='NAC')
        else:
            base_conv2d_block = functools.partial(Conv2dBlock, kernel_size=kernel_size, stride=1, padding=padding, bias=True, weight_norm_type=weight_norm_type, nonlinearity=nonlinearity, order='NAC')
        in_num_labels = num_labels
        in_num_labels += 2 if self.use_posenc_in_input_layer else 0
        self.head_0 = Conv2dBlock(in_num_labels, 8 * num_filters, kernel_size=kernel_size, stride=1, padding=padding, weight_norm_type=weight_norm_type, activation_norm_type='none', nonlinearity=nonlinearity)
        if self.use_style_encoder:
            self.cbn_head_0 = base_cbn2d_block(8 * num_filters, 16 * num_filters)
        else:
            self.conv_head_0 = base_conv2d_block(8 * num_filters, 16 * num_filters)
        self.head_1 = base_res2d_block(16 * num_filters, 16 * num_filters)
        self.head_2 = base_res2d_block(16 * num_filters, 16 * num_filters)
        self.up_0a = base_res2d_block(16 * num_filters, 8 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_0a = base_cbn2d_block(8 * num_filters, 8 * num_filters)
        else:
            self.conv_up_0a = base_conv2d_block(8 * num_filters, 8 * num_filters)
        self.up_0b = base_res2d_block(8 * num_filters, 8 * num_filters)
        self.up_1a = base_res2d_block(8 * num_filters, 4 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_1a = base_cbn2d_block(4 * num_filters, 4 * num_filters)
        else:
            self.conv_up_1a = base_conv2d_block(4 * num_filters, 4 * num_filters)
        self.up_1b = base_res2d_block(4 * num_filters, 4 * num_filters)
        self.up_2a = base_res2d_block(4 * num_filters, 4 * num_filters)
        if self.use_style_encoder:
            self.cbn_up_2a = base_cbn2d_block(4 * num_filters, 4 * num_filters)
        else:
            self.conv_up_2a = base_conv2d_block(4 * num_filters, 4 * num_filters)
        self.up_2b = base_res2d_block(4 * num_filters, 2 * num_filters)
        self.conv_img256 = Conv2dBlock(2 * num_filters, image_channels, 5, stride=1, padding=2, weight_norm_type=weight_norm_type, activation_norm_type='none', nonlinearity=nonlinearity, order='ANC')
        self.base = 16
        if self.out_image_small_side_size == 512:
            self.up_3a = base_res2d_block(2 * num_filters, 1 * num_filters)
            self.up_3b = base_res2d_block(1 * num_filters, 1 * num_filters)
            self.conv_img512 = Conv2dBlock(1 * num_filters, image_channels, 5, stride=1, padding=2, weight_norm_type=weight_norm_type, activation_norm_type='none', nonlinearity=nonlinearity, order='ANC')
            self.base = 32
        if self.out_image_small_side_size == 1024:
            self.up_3a = base_res2d_block(2 * num_filters, 1 * num_filters)
            self.up_3b = base_res2d_block(1 * num_filters, 1 * num_filters)
            self.conv_img512 = Conv2dBlock(1 * num_filters, image_channels, 5, stride=1, padding=2, weight_norm_type=weight_norm_type, activation_norm_type='none', nonlinearity=nonlinearity, order='ANC')
            self.up_4a = base_res2d_block(num_filters, num_filters // 2)
            self.up_4b = base_res2d_block(num_filters // 2, num_filters // 2)
            self.conv_img1024 = Conv2dBlock(num_filters // 2, image_channels, 5, stride=1, padding=2, weight_norm_type=weight_norm_type, activation_norm_type='none', nonlinearity=nonlinearity, order='ANC')
            self.nearest_upsample4x = NearestUpsample(scale_factor=4, mode='nearest')
            self.base = 64
        if self.out_image_small_side_size != 256 and self.out_image_small_side_size != 512 and self.out_image_small_side_size != 1024:
            raise ValueError('Generation image size (%d, %d) not supported' % (self.out_image_small_side_size, self.out_image_small_side_size))
        self.nearest_upsample2x = NearestUpsample(scale_factor=2, mode='nearest')
        xv, yv = torch.meshgrid([torch.arange(-1, 1.1, 2.0 / 15), torch.arange(-1, 1.1, 2.0 / 15)])
        self.xy = torch.cat((xv.unsqueeze(0), yv.unsqueeze(0)), 0).unsqueeze(0)
        self.xy = self.xy

    def forward(self, data):
        """SPADE Generator forward.

        Args:
            data (dict):
              - data  (N x C1 x H x W tensor) : Ground truth images.
              - label (N x C2 x H x W tensor) : Semantic representations.
              - z (N x style_dims tensor): Gaussian random noise.
        Returns:
            output (dict):
              - fake_images (N x 3 x H x W tensor): Fake images.
        """
        seg = data['label']
        if self.use_style_encoder:
            z = data['z']
            z = self.fc_0(z)
            z = self.fc_1(z)
        sy = math.floor(seg.size()[2] * 1.0 / self.base)
        sx = math.floor(seg.size()[3] * 1.0 / self.base)
        in_seg = F.interpolate(seg, size=[sy, sx], mode='nearest')
        if self.use_posenc_in_input_layer:
            in_xy = F.interpolate(self.xy, size=[sy, sx], mode='bicubic')
            in_seg_xy = torch.cat((in_seg, in_xy.expand(in_seg.size()[0], 2, sy, sx)), 1)
        else:
            in_seg_xy = in_seg
        x = self.head_0(in_seg_xy)
        if self.use_style_encoder:
            x = self.cbn_head_0(x, z)
        else:
            x = self.conv_head_0(x)
        x = self.head_1(x, seg)
        x = self.head_2(x, seg)
        x = self.nearest_upsample2x(x)
        x = self.up_0a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_0a(x, z)
        else:
            x = self.conv_up_0a(x)
        x = self.up_0b(x, seg)
        x = self.nearest_upsample2x(x)
        x = self.up_1a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_1a(x, z)
        else:
            x = self.conv_up_1a(x)
        x = self.up_1b(x, seg)
        x = self.nearest_upsample2x(x)
        x = self.up_2a(x, seg)
        if self.use_style_encoder:
            x = self.cbn_up_2a(x, z)
        else:
            x = self.conv_up_2a(x)
        x = self.up_2b(x, seg)
        x = self.nearest_upsample2x(x)
        if self.out_image_small_side_size == 256:
            x256 = self.conv_img256(x)
            x = torch.tanh(self.output_multiplier * x256)
        elif self.out_image_small_side_size == 512:
            x256 = self.conv_img256(x)
            x256 = self.nearest_upsample2x(x256)
            x = self.up_3a(x, seg)
            x = self.up_3b(x, seg)
            x = self.nearest_upsample2x(x)
            x512 = self.conv_img512(x)
            x = torch.tanh(self.output_multiplier * (x256 + x512))
        elif self.out_image_small_side_size == 1024:
            x256 = self.conv_img256(x)
            x256 = self.nearest_upsample4x(x256)
            x = self.up_3a(x, seg)
            x = self.up_3b(x, seg)
            x = self.nearest_upsample2x(x)
            x512 = self.conv_img512(x)
            x512 = self.nearest_upsample2x(x512)
            x = self.up_4a(x, seg)
            x = self.up_4b(x, seg)
            x = self.nearest_upsample2x(x)
            x1024 = self.conv_img1024(x)
            x = torch.tanh(self.output_multiplier * (x256 + x512 + x1024))
        output = dict()
        output['fake_images'] = x
        return output


class DualAdaptiveNorm(nn.Module):

    def __init__(self, num_features, cond_dims, projection_bias=True, weight_norm_type='', activation_norm_type='instance', activation_norm_params=None, apply_noise=False, bias_only=False, init_gain=1.0, fc_scale=None, is_spatial=None):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        self.bias_only = bias_only
        if type(cond_dims) != list:
            cond_dims = [cond_dims]
        if is_spatial is None:
            is_spatial = [(False) for _ in range(len(cond_dims))]
        self.is_spatial = is_spatial
        for cond_dim, this_is_spatial in zip(cond_dims, is_spatial):
            kwargs = dict(weight_norm_type=weight_norm_type, bias=projection_bias, init_gain=init_gain, output_scale=fc_scale)
            if this_is_spatial:
                self.gammas.append(Conv2dBlock(cond_dim, num_features, 1, 1, 0, **kwargs))
                self.betas.append(Conv2dBlock(cond_dim, num_features, 1, 1, 0, **kwargs))
            else:
                self.gammas.append(LinearBlock(cond_dim, num_features, **kwargs))
                self.betas.append(LinearBlock(cond_dim, num_features, **kwargs))
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, 2, **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **_kwargs):
        assert len(cond_inputs) == len(self.gammas)
        output = self.norm(x) if self.norm is not None else x
        for cond, gamma_layer, beta_layer in zip(cond_inputs, self.gammas, self.betas):
            if cond is None:
                continue
            gamma = gamma_layer(cond)
            beta = beta_layer(cond)
            if cond.dim() == 4 and gamma.shape != x.shape:
                gamma = F.interpolate(gamma, size=x.size()[2:], mode='bilinear')
                beta = F.interpolate(beta, size=x.size()[2:], mode='bilinear')
            elif cond.dim() == 2:
                gamma = gamma[:, :, None, None]
                beta = beta[:, :, None, None]
            if self.bias_only:
                output = output + beta
            else:
                output = output * (1 + gamma) + beta
        return output


class HyperConv2d(nn.Module):
    """Hyper Conv2d initialization.

    Args:
        in_channels (int): Dummy parameter.
        out_channels (int): Dummy parameter.
        kernel_size (int or tuple): Dummy parameter.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution. Default: 1
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        padding_mode (string, optional, default='zeros'):
            ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True): If ``True``,
            adds a learnable bias to the output.
    """

    def __init__(self, in_channels=0, out_channels=0, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.conditional = True

    def forward(self, x, *args, conv_weights=(None, None), **kwargs):
        """Hyper Conv2d forward. Convolve x using the provided weight and bias.

        Args:
            x (N x C x H x W tensor): Input tensor.
            conv_weights (N x C2 x C1 x k x k tensor or list of tensors):
                Convolution weights or [weight, bias].
        Returns:
            y (N x C2 x H x W tensor): Output tensor.
        """
        if conv_weights is None:
            conv_weight, conv_bias = None, None
        elif isinstance(conv_weights, torch.Tensor):
            conv_weight, conv_bias = conv_weights, None
        else:
            conv_weight, conv_bias = conv_weights
        if conv_weight is None:
            return x
        if conv_bias is None:
            if self.use_bias:
                raise ValueError('bias not provided but set to true during initialization')
            conv_bias = [None] * x.size(0)
        if self.padding_mode != 'zeros':
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        y = None
        for i in range(x.size(0)):
            if self.stride >= 1:
                yi = F.conv2d(x[i:i + 1], weight=conv_weight[i], bias=conv_bias[i], stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups)
            else:
                yi = F.conv_transpose2d(x[i:i + 1], weight=conv_weight[i], bias=conv_bias[i], padding=self.padding, stride=int(1 / self.stride), dilation=self.dilation, output_padding=self.padding, groups=self.groups)
            y = torch.cat([y, yi]) if y is not None else yi
        return y


class HyperSpatiallyAdaptiveNorm(nn.Module):
    """Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the conditional input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        is_hyper (bool): Whether to use hyper SPADE.
    """

    def __init__(self, num_features, cond_dims, num_filters=0, kernel_size=3, weight_norm_type='', activation_norm_type='sync_batch', is_hyper=True):
        super().__init__()
        padding = kernel_size // 2
        self.mlps = nn.ModuleList()
        if type(cond_dims) != list:
            cond_dims = [cond_dims]
        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if not is_hyper or i != 0:
                if num_filters > 0:
                    mlp += [Conv2dBlock(cond_dim, num_filters, kernel_size, padding=padding, weight_norm_type=weight_norm_type, nonlinearity='relu')]
                mlp_ch = cond_dim if num_filters == 0 else num_filters
                mlp += [Conv2dBlock(mlp_ch, num_features * 2, kernel_size, padding=padding, weight_norm_type=weight_norm_type)]
                mlp = nn.Sequential(*mlp)
            else:
                if num_filters > 0:
                    raise ValueError('Multi hyper layer not supported yet.')
                mlp = HyperConv2d(padding=padding)
            self.mlps.append(mlp)
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, 2, affine=False)
        self.conditional = True

    def forward(self, x, *cond_inputs, norm_weights=(None, None), **_kwargs):
        """Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (4D tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
            norm_weights (5D tensor or list of tensors): conv weights or
            [weights, biases].
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x)
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            if type(cond_inputs[i]) == list:
                cond_input, mask = cond_inputs[i]
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)
            else:
                cond_input = cond_inputs[i]
                mask = None
            label_map = F.interpolate(cond_input, size=x.size()[2:])
            if norm_weights is None or norm_weights[0] is None or i != 0:
                affine_params = self.mlps[i](label_map)
            else:
                affine_params = self.mlps[i](label_map, conv_weights=norm_weights)
            gamma, beta = affine_params.chunk(2, dim=1)
            if mask is not None:
                gamma = gamma * (1 - mask)
                beta = beta * (1 - mask)
            output = output * (1 + gamma) + beta
        return output


class LayerNorm2d(nn.Module):
    """Layer Normalization as introduced in
    https://arxiv.org/abs/1607.06450.
    This is the usual way to apply layer normalization in CNNs.
    Note that unlike the pytorch implementation which applies per-element
    scale and bias, here it applies per-channel scale and bias, similar to
    batch/instance normalization.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional, default=1e-5): a value added to the
            denominator for numerical stability.
        affine (bool, optional, default=False): If ``True``, performs
            affine transformation after normalization.
    """

    def __init__(self, num_features, eps=1e-05, channel_only=False, affine=True):
        super(LayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.channel_only = channel_only
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).fill_(1.0))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensor.
        """
        shape = [-1] + [1] * (x.dim() - 1)
        if self.channel_only:
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class PixelLayerNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            return self.norm(x.permute(0, 2, 3, 1).view(-1, c)).view(b, h, w, c).permute(0, 3, 1, 2)
        else:
            return self.norm(x)


class ScaleNorm(nn.Module):
    """Scale normalization:
    "Transformers without Tears: Improving the Normalization of Self-Attention"
    Modified from:
    https://github.com/tnq177/transformers_without_tears
    """

    def __init__(self, dim=-1, learned_scale=True, eps=1e-05):
        super().__init__()
        if learned_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        self.dim = dim
        self.eps = eps
        self.learned_scale = learned_scale

    def forward(self, x):
        scale = self.scale * torch.rsqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x * scale

    def extra_repr(self):
        s = 'learned_scale={learned_scale}'
        return s.format(**self.__dict__)


class PixelNorm(ScaleNorm):

    def __init__(self, learned_scale=False, eps=1e-05, **_kwargs):
        super().__init__(1, learned_scale, eps)


class PartialConv2d(nn.Conv2d):
    """Partial 2D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3]
        self.last_size = None, None, None, None
        self.update_mask = None
        self.mask_ratio = None
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        """

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
        """
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater
                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0], x.data.shape[1], x.data.shape[2], x.data.shape[3])
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3])
                else:
                    mask = mask_in
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                eps = 1e-06
                self.mask_ratio = self.slide_winsize / (self.update_mask + eps)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv2d, self).forward(torch.mul(x, mask) if mask_in is not None else x)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv3d(nn.Conv3d):
    """Partial 3D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv3d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        self.weight_maskUpdater = self.weight_maskUpdater
        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3] * shape[4]
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        """

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``, it
                masks the valid input region.
        """
        assert len(x.shape) == 5
        with torch.no_grad():
            mask = mask_in
            update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            mask_ratio = self.slide_winsize / (update_mask + 1e-08)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)
        raw_out = super(PartialConv3d, self).forward(torch.mul(x, mask_in))
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            if mask_in is not None:
                output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)
        if self.return_mask:
            return output, update_mask
        else:
            return output


class _BasePartialConvBlock(_BaseConvBlock):
    """An abstract wrapper class that wraps a partial convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, order, input_dim, clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, False, order, input_dim, clamp, blur_kernel, output_scale, init_gain)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        if input_dim == 2:
            layer_type = PartialConv2d
        elif input_dim == 3:
            layer_type = PartialConv3d
        else:
            raise ValueError('Partial conv only supports 2D and 3D conv now.')
        layer = layer_type(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, multi_channel=self.multi_channel, return_mask=self.return_mask)
        return layer

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        mask_out = None
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            elif getattr(layer, 'partial_conv', False):
                x = layer(x, mask_in=mask_in, **kw_cond_inputs)
                if type(x) == tuple:
                    x, mask_out = x
            else:
                x = layer(x)
        if mask_out is not None:
            return x, mask_out
        return x


class PartialConv2dBlock(_BasePartialConvBlock):
    """A Wrapper class that wraps ``PartialConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, multi_channel=False, return_mask=True, apply_noise=False, order='CNA', clamp=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, order, 2, clamp)


class PartialSequential(nn.Sequential):
    """Sequential block for partial convolutions."""

    def __init__(self, *modules):
        super(PartialSequential, self).__init__(*modules)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensor.
        """
        act = x[:, :-1]
        mask = x[:, -1].unsqueeze(1)
        for module in self:
            act, mask = module(act, mask_in=mask)
        return act


class SpatiallyAdaptiveNorm(nn.Module):
    """Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self, num_features, cond_dims, num_filters=128, kernel_size=3, weight_norm_type='', separate_projection=False, activation_norm_type='sync_batch', activation_norm_params=None, bias_only=False, partial=False, interpolation='nearest'):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        self.bias_only = bias_only
        self.interpolation = interpolation
        if type(cond_dims) != list:
            cond_dims = [cond_dims]
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)
        if not isinstance(partial, list):
            partial = [partial] * len(cond_dims)
        else:
            assert len(partial) >= len(cond_dims)
        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            conv_block = PartialConv2dBlock if partial[i] else Conv2dBlock
            sequential = PartialSequential if partial[i] else nn.Sequential
            if num_filters[i] > 0:
                mlp += [conv_block(cond_dim, num_filters[i], kernel_size, padding=padding, weight_norm_type=weight_norm_type, nonlinearity='relu')]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]
            if self.separate_projection:
                if partial[i]:
                    raise NotImplementedError('Separate projection not yet implemented for ' + 'partial conv')
                self.mlps.append(nn.Sequential(*mlp))
                self.gammas.append(conv_block(mlp_ch, num_features, kernel_size, padding=padding, weight_norm_type=weight_norm_type))
                self.betas.append(conv_block(mlp_ch, num_features, kernel_size, padding=padding, weight_norm_type=weight_norm_type))
            else:
                mlp += [conv_block(mlp_ch, num_features * 2, kernel_size, padding=padding, weight_norm_type=weight_norm_type)]
                self.mlps.append(sequential(*mlp))
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, 2, **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **_kwargs):
        """Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (N x C1 x H x W tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:], mode=self.interpolation)
            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = affine_params.chunk(2, dim=1)
            if self.bias_only:
                output = output + beta
            else:
                output = output * (1 + gamma) + beta
        return output


def get_activation_norm_layer(num_features, norm_type, input_dim, **norm_params):
    """Return an activation normalization layer.

    Args:
        num_features (int): Number of feature channels.
        norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        input_dim (int): Number of input dimensions.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the activation normalization.
    """
    input_dim = max(input_dim, 1)
    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'batch':
        norm = getattr(nn, 'BatchNorm%dd' % input_dim)
        norm_layer = norm(num_features, **norm_params)
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)
        norm = getattr(nn, 'InstanceNorm%dd' % input_dim)
        norm_layer = norm(num_features, affine=affine, **norm_params)
    elif norm_type == 'sync_batch':
        norm_layer = SyncBatchNorm(num_features, **norm_params)
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        norm_layer = LayerNorm2d(num_features, **norm_params)
    elif norm_type == 'pixel_layer':
        elementwise_affine = norm_params.pop('affine', True)
        norm_layer = PixelLayerNorm(num_features, elementwise_affine=elementwise_affine, **norm_params)
    elif norm_type == 'scale':
        norm_layer = ScaleNorm(**norm_params)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm(**norm_params)
        if imaginaire.config.USE_JIT:
            norm_layer = torch.jit.script(norm_layer)
    elif norm_type == 'group':
        num_groups = norm_params.pop('num_groups', 4)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=num_groups, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'dual_adaptive':
        norm_layer = DualAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers only supports 2D input')
        norm_layer = SpatiallyAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'hyper_spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers only supports 2D input')
        norm_layer = HyperSpatiallyAdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError('Activation norm layer %s is not recognized' % norm_type)
    return norm_layer


class AdaptiveNorm(nn.Module):
    """Adaptive normalization layer. The layer first normalizes the input, then
    performs an affine transformation using parameters computed from the
    conditional inputs.

    Args:
        num_features (int): Number of channels in the input tensor.
        cond_dims (int): Number of channels in the conditional inputs.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``, or ``'weight_demod'``.
        projection (bool): If ``True``, project the conditional input to gamma
            and beta using a fully connected layer, otherwise directly use
            the conditional input as gamma and beta.
        projection_bias (bool) If ``True``, use bias in the fully connected
            projection layer.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        input_dim (int): Number of dimensions of the input tensor.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self, num_features, cond_dims, weight_norm_type='', projection=True, projection_bias=True, separate_projection=False, input_dim=2, activation_norm_type='instance', activation_norm_params=None, apply_noise=False, add_bias=True, input_scale=1.0, init_gain=1.0):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, input_dim, **vars(activation_norm_params))
        if apply_noise:
            self.noise_layer = ApplyNoise()
        else:
            self.noise_layer = None
        if projection:
            if separate_projection:
                self.fc_gamma = LinearBlock(cond_dims, num_features, weight_norm_type=weight_norm_type, bias=projection_bias)
                self.fc_beta = LinearBlock(cond_dims, num_features, weight_norm_type=weight_norm_type, bias=projection_bias)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2, weight_norm_type=weight_norm_type, bias=projection_bias)
        self.projection = projection
        self.separate_projection = separate_projection
        self.input_scale = input_scale
        self.add_bias = add_bias
        self.conditional = True
        self.init_gain = init_gain

    def forward(self, x, y, noise=None, **_kwargs):
        """Adaptive Normalization forward.

        Args:
            x (N x C1 x * tensor): Input tensor.
            y (N x C2 tensor): Conditional information.
        Returns:
            out (N x C1 x * tensor): Output tensor.
        """
        y = y * self.input_scale
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(x.dim() - gamma.dim()):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
            else:
                y = self.fc(y)
                for _ in range(x.dim() - y.dim()):
                    y = y.unsqueeze(-1)
                gamma, beta = y.chunk(2, 1)
        else:
            for _ in range(x.dim() - y.dim()):
                y = y.unsqueeze(-1)
            gamma, beta = y.chunk(2, 1)
        if self.norm is not None:
            x = self.norm(x)
        if self.noise_layer is not None:
            x = self.noise_layer(x, noise=noise)
        if self.add_bias:
            x = torch.addcmul(beta, x, 1 + gamma)
            return x
        else:
            return x * (1 + gamma), beta.squeeze(3).squeeze(2)


class SplitMeanStd(nn.Module):

    def __init__(self, num_features, eps=1e-05, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.multiple_outputs = True

    def forward(self, x):
        b, c, h, w = x.size()
        mean = x.view(b, c, -1).mean(-1)[:, :, None, None]
        var = x.view(b, c, -1).var(-1)[:, :, None, None]
        std = torch.sqrt(var + self.eps)
        return x, torch.cat((mean, std), dim=1)


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, demodulate=True, eps=1e-08):
        assert dilation == 1 and groups == 1
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.demodulate = demodulate
        self.conditional = True

    def forward(self, x, style, **_kwargs):
        batch, in_channel, height, width = x.shape
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.weight.unsqueeze(0) * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = self.bias.repeat(batch)
        else:
            bias = self.bias
        x = x.view(1, batch * in_channel, height, width)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0, 0
        else:
            padding = self.padding
        if self.stride == 0.5:
            weight = weight.view(batch, self.out_channels, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, bias, padding=padding, stride=2, groups=batch)
        elif self.stride == 2:
            out = F.conv2d(x, weight, bias, padding=padding, stride=2, groups=batch)
        else:
            out = F.conv2d(x, weight, bias, padding=padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        return out

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class ModulatedConv2dBlock(_BaseConvBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=True, blur=True, order='CNA', demodulate=True, eps=True, style_dim=None, clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        self.eps = eps
        self.demodulate = demodulate
        assert style_dim is not None
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, 2, clamp, blur_kernel, output_scale, init_gain)
        self.modulation = LinearBlock(style_dim, in_channels, weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        assert input_dim == 2
        layer = ModulatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, self.demodulate, self.eps)
        return layer

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                assert len(cond_inputs) == 1
                style = cond_inputs[0]
                x = layer(x, self.modulation(style), **kw_cond_inputs)
            else:
                x = layer(x)
            if self.clamp is not None and isinstance(layer, ModulatedConv2d):
                x.clamp_(max=self.clamp)
        return x

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and self.weight_norm_type != '':
                mod_str = mod_str[:-1] + ', weight_norm={}'.format(self.weight_norm_type) + ', demodulate={}'.format(self.demodulate) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        child_lines.append(self._addindent('Modulation(' + repr(self.modulation) + ')', 2))
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str


class EmbeddingBlock(_BaseConvBlock):

    def __init__(self, in_features, out_features, bias=True, weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, order='CNA', clamp=None, output_scale=None, init_gain=1.0, **_kwargs):
        if bool(_kwargs):
            warnings.warn(f'Unused keyword arguments {_kwargs}')
        super().__init__(in_features, out_features, None, None, None, None, None, bias, None, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, False, order, 0, clamp, None, output_scale, init_gain)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        assert input_dim == 0
        return nn.Embedding(in_channels, out_channels)


class Embedding2d(nn.Embedding):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        return F.embedding(x.squeeze(1).long(), self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).permute(0, 3, 1, 2).contiguous()


class Embedding2dBlock(_BaseConvBlock):

    def __init__(self, in_features, out_features, bias=True, weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, order='CNA', clamp=None, output_scale=None, init_gain=1.0, **_kwargs):
        if bool(_kwargs):
            warnings.warn(f'Unused keyword arguments {_kwargs}')
        super().__init__(in_features, out_features, None, None, None, None, None, bias, None, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, False, order, 0, clamp, None, output_scale, init_gain)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        assert input_dim == 0
        return Embedding2d(in_channels, out_channels)


class Conv1dBlock(_BaseConvBlock):
    """A Wrapper class that wraps ``torch.nn.Conv1d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, blur=False, order='CNA', clamp=None, output_scale=None, init_gain=1.0, **_kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, 1, clamp, None, output_scale, init_gain)


class Conv3dBlock(_BaseConvBlock):
    """A Wrapper class that wraps ``torch.nn.Conv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, blur=False, order='CNA', clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, 3, clamp, blur_kernel, output_scale, init_gain)


class PartialConv3dBlock(_BasePartialConvBlock):
    """A Wrapper class that wraps ``PartialConv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, multi_channel=False, return_mask=True, apply_noise=False, order='CNA', clamp=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, order, 3, clamp)


class _MultiOutBaseConvBlock(_BaseConvBlock):
    """An abstract wrapper class that wraps a hyper convolutional layer with
    normalization and nonlinearity. It can return multiple outputs, if some
    layers in the block return more than one output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, input_dim, clamp=None, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, input_dim, clamp, blur_kernel, output_scale, init_gain)
        self.multiple_outputs = True

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Main output tensor.
              - other_outputs (list of tensors): Other output tensors.
        """
        other_outputs = []
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            if getattr(layer, 'multiple_outputs', False):
                x, other_output = layer(x)
                other_outputs.append(other_output)
            else:
                x = layer(x)
        return x, *other_outputs


class MultiOutConv2dBlock(_MultiOutBaseConvBlock):
    """A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity. It can return multiple outputs, if some layers in the block
    return more than one output.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or float or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, blur=False, order='CNA', clamp=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, 2, clamp)


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            h, w = size, size
        else:
            h, w = size
        self.input = nn.Parameter(torch.randn(1, channel, h, w))

    def forward(self):
        return self.input


class NonLocal2dBlock(nn.Module):
    """Self attention Layer

    Args:
        in_channels (int): Number of channels in the input tensor.
        scale (bool, optional, default=True): If ``True``, scale the
            output by a learnable parameter.
        clamp (bool, optional, default=``False``): If ``True``, clamp the
            scaling parameter to (-1, 1).
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, weight_norm_params.__dict__ will be used as
            keyword arguments when initializing weight normalization.
        bias (bool, optional, default=True): If ``True``, adds bias in the
            convolutional blocks.
    """

    def __init__(self, in_channels, scale=True, clamp=False, weight_norm_type='none', weight_norm_params=None, bias=True):
        super(NonLocal2dBlock, self).__init__()
        self.clamp = clamp
        self.gamma = nn.Parameter(torch.zeros(1)) if scale else 1.0
        self.in_channels = in_channels
        base_conv2d_block = partial(Conv2dBlock, kernel_size=1, stride=1, padding=0, weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params, bias=bias)
        self.theta = base_conv2d_block(in_channels, in_channels // 8)
        self.phi = base_conv2d_block(in_channels, in_channels // 8)
        self.g = base_conv2d_block(in_channels, in_channels // 2)
        self.out_conv = base_conv2d_block(in_channels // 2, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        """

        Args:
            x (tensor) : input feature maps (B X C X W X H)
        Returns:
            (tuple):
              - out (tensor) : self attention value + input feature
              - attention (tensor): B x N x N (N is Width*Height)
        """
        n, c, h, w = x.size()
        theta = self.theta(x).view(n, -1, h * w).permute(0, 2, 1)
        phi = self.phi(x)
        phi = self.max_pool(phi).view(n, -1, h * w // 4)
        energy = torch.bmm(theta, phi)
        attention = self.softmax(energy)
        g = self.g(x)
        g = self.max_pool(g).view(n, -1, h * w // 4)
        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(n, c // 2, h, w)
        out = self.out_conv(out)
        if self.clamp:
            out = self.gamma.clamp(-1, 1) * out + x
        else:
            out = self.gamma * out + x
        return out


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2, scale=2 ** 0.5, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope, inplace=self.inplace) * self.scale


class ModulatedRes2dBlock(_BaseResBlock):

    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=True, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1, demodulate=True, eps=1e-08):
        block = functools.partial(ModulatedConv2dBlock, style_dim=style_dim, demodulate=demodulate, eps=eps)
        skip_block = Conv2dBlock
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale, skip_block=skip_block)

    def conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        assert len(list(cond_inputs)) == 2
        dx = self.conv_block_0(x, cond_inputs[0], **kw_cond_inputs)
        dx = self.conv_block_1(dx, cond_inputs[1], **kw_cond_inputs)
        return dx


class ResLinearBlock(_BaseResBlock):
    """Residual block with full-connected layers.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, None, 1, None, None, None, bias, None, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, LinearBlock, learn_shortcut, clamp, output_scale)


class Res1dBlock(_BaseResBlock):
    """Residual block for 1D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv1dBlock, learn_shortcut, clamp, output_scale)


class Res3dBlock(_BaseResBlock):
    """Residual block for 3D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv3dBlock, learn_shortcut, clamp, output_scale)


class _BaseHyperResBlock(_BaseResBlock):
    """An abstract class for hyper residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, is_hyper_conv, is_hyper_norm, block, learn_shortcut, clamp=None, output_scale=1):
        block = functools.partial(block, is_hyper_conv=is_hyper_conv, is_hyper_norm=is_hyper_norm)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale)

    def forward(self, x, *cond_inputs, conv_weights=(None,) * 3, norm_weights=(None,) * 3, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            conv_weights (list of tensors): Convolution weights for
                three convolutional layers respectively.
            norm_weights (list of tensors): Normalization weights for
                three convolutional layers respectively.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, conv_weights=conv_weights[0], norm_weights=norm_weights[0])
        dx = self.conv_block_1(dx, *cond_inputs, conv_weights=conv_weights[1], norm_weights=norm_weights[1])
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, conv_weights=conv_weights[2], norm_weights=norm_weights[2])
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return self.output_scale * output


class HyperRes2dBlock(_BaseHyperResBlock):
    """Hyper residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='', weight_norm_params=None, activation_norm_type='', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', is_hyper_conv=False, is_hyper_norm=False, learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, is_hyper_conv, is_hyper_norm, HyperConv2dBlock, learn_shortcut, clamp, output_scale)


class _BaseDownResBlock(_BaseResBlock):
    """An abstract class for residual blocks with downsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, pooling, down_factor, learn_shortcut, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale)
        self.pooling = pooling(down_factor)

    def forward(self, x, *cond_inputs):
        """

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs)
        dx = self.pooling(dx)
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        x_shortcut = self.pooling(x_shortcut)
        output = x_shortcut + dx
        return self.output_scale * output


class DownRes2dBlock(_BaseDownResBlock):
    """Residual block for 2D input with downsampling.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        pooling (class, optional, default=nn.AvgPool2d): Pytorch pooling
            layer to be used.
        down_factor (int, optional, default=2): Downsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', pooling=nn.AvgPool2d, down_factor=2, learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv2dBlock, pooling, down_factor, learn_shortcut, clamp, output_scale)


class _BaseUpResBlock(_BaseResBlock):
    """An abstract class for residual blocks with upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, upsample, up_factor, learn_shortcut, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale)
        self.order = order
        self.upsample = upsample(scale_factor=up_factor)

    def _get_stride_blur(self):
        first_stride, second_stride = self.stride, 1
        first_blur, second_blur = self.blur, False
        shortcut_blur = False
        shortcut_stride = 1
        if self.blur:
            self.upsample = BlurUpsample()
        else:
            shortcut_stride = self.stride
            self.upsample = nn.Upsample(scale_factor=2)
        return first_stride, second_stride, shortcut_stride, first_blur, second_blur, shortcut_blur

    def forward(self, x, *cond_inputs):
        """Implementation of the up residual block forward function.
        If the order is 'NAC' for the first residual block, we will first
        do the activation norm and nonlinearity, in the original resolution.
        We will then upsample the activation map to a higher resolution. We
        then do the convolution.
        It is is other orders, then we first do the whole processing and
        then upsample.

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        if self.learn_shortcut:
            x_shortcut = self.upsample(x)
            x_shortcut = self.conv_block_s(x_shortcut, *cond_inputs)
        else:
            x_shortcut = self.upsample(x)
        if self.order[0:3] == 'NAC':
            for ix, layer in enumerate(self.conv_block_0.layers.values()):
                if getattr(layer, 'conditional', False):
                    x = layer(x, *cond_inputs)
                else:
                    x = layer(x)
                if ix == 1:
                    x = self.upsample(x)
        else:
            x = self.conv_block_0(x, *cond_inputs)
            x = self.upsample(x)
        x = self.conv_block_1(x, *cond_inputs)
        output = x_shortcut + x
        return self.output_scale * output


class UpRes2dBlock(_BaseUpResBlock):
    """Residual block for 2D input with downsampling.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        upsample (class, optional, default=NearestUpsample): PPytorch
            upsampling layer to be used.
        up_factor (int, optional, default=2): Upsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', upsample=NearestUpsample, up_factor=2, learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv2dBlock, upsample, up_factor, learn_shortcut, clamp, output_scale)


class _BasePartialResBlock(_BaseResBlock):
    """An abstract class for residual blocks with partial convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp=None, output_scale=1):
        block = functools.partial(block, multi_channel=multi_channel, return_mask=return_mask)
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale)

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - output (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        if self.conv_block_0.layers.conv.return_mask:
            dx, mask_out = self.conv_block_0(x, *cond_inputs, mask_in=mask_in, **kw_cond_inputs)
            dx, mask_out = self.conv_block_1(dx, *cond_inputs, mask_in=mask_out, **kw_cond_inputs)
        else:
            dx = self.conv_block_0(x, *cond_inputs, mask_in=mask_in, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs, mask_in=mask_in, **kw_cond_inputs)
            mask_out = None
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, mask_in=mask_in, **kw_cond_inputs)
            if type(x_shortcut) == tuple:
                x_shortcut, _ = x_shortcut
        else:
            x_shortcut = x
        output = x_shortcut + dx
        if mask_out is not None:
            return output, mask_out
        return self.output_scale * output


class PartialRes2dBlock(_BasePartialResBlock):
    """Residual block for 2D input with partial convolution.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, multi_channel=False, return_mask=True, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, hidden_channels_equal_out_channels, order, PartialConv2dBlock, learn_shortcut, clamp, output_scale)


class PartialRes3dBlock(_BasePartialResBlock):
    """Residual block for 3D input with partial convolution.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, multi_channel=False, return_mask=True, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, hidden_channels_equal_out_channels, order, PartialConv3dBlock, learn_shortcut, clamp, output_scale)


class _BaseMultiOutResBlock(_BaseResBlock):
    """An abstract class for residual blocks that can returns multiple outputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp=None, output_scale=1, blur=False, upsample_first=True):
        self.multiple_outputs = True
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, clamp, output_scale, blur=blur, upsample_first=upsample_first)

    def forward(self, x, *cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
        Returns:
            (tuple):
              - output (tensor): Output tensor.
              - aux_outputs_0 (tensor): Auxiliary output of the first block.
              - aux_outputs_1 (tensor): Auxiliary output of the second block.
        """
        dx, aux_outputs_0 = self.conv_block_0(x, *cond_inputs)
        dx, aux_outputs_1 = self.conv_block_1(dx, *cond_inputs)
        if self.learn_shortcut:
            x_shortcut, _ = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return self.output_scale * output, aux_outputs_0, aux_outputs_1


class MultiOutRes2dBlock(_BaseMultiOutResBlock):
    """Residual block for 2D input. It can return multiple outputs, if some
    layers in the block return more than one output.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=None, clamp=None, output_scale=1, blur=False, upsample_first=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, MultiOutConv2dBlock, learn_shortcut, clamp, output_scale, blur=blur, upsample_first=upsample_first)


class _BaseDeepResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut, output_scale, skip_block=None, blur=True, border_free=True, resample_first=True, skip_weight_norm=True, hidden_channel_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale = output_scale
        self.resample_first = resample_first
        self.stride = stride
        self.blur = blur
        self.border_free = border_free
        assert not border_free
        if skip_block is None:
            skip_block = block
        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        self.learn_shortcut = learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        hidden_channels = in_channels // hidden_channel_ratio
        residual_params = {}
        shortcut_params = {}
        base_params = dict(dilation=dilation, groups=groups, padding_mode=padding_mode)
        residual_params.update(base_params)
        residual_params.update(dict(activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params, weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params, apply_noise=apply_noise))
        shortcut_params.update(base_params)
        shortcut_params.update(dict(kernel_size=1))
        if skip_activation_norm:
            shortcut_params.update(dict(activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params, apply_noise=False))
        if skip_weight_norm:
            shortcut_params.update(dict(weight_norm_type=weight_norm_type, weight_norm_params=weight_norm_params))
        if order.find('A') < order.find('C') and (activation_norm_type == '' or activation_norm_type == 'none'):
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity
        first_stride, second_stride, shortcut_stride, first_blur, second_blur, shortcut_blur = self._get_stride_blur()
        self.conv_block_1x1_in = block(in_channels, hidden_channels, 1, 1, 0, bias=biases[0], nonlinearity=nonlinearity, order=order[0:3], inplace_nonlinearity=first_inplace, **residual_params)
        self.conv_block_0 = block(hidden_channels, hidden_channels, kernel_size=2 if self.border_free and first_stride < 1 else kernel_size, padding=padding, bias=biases[0], nonlinearity=nonlinearity, order=order[0:3], inplace_nonlinearity=inplace_nonlinearity, stride=first_stride, blur=first_blur, **residual_params)
        self.conv_block_1 = block(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=biases[1], nonlinearity=nonlinearity, order=order[3:], inplace_nonlinearity=inplace_nonlinearity, stride=second_stride, blur=second_blur, **residual_params)
        self.conv_block_1x1_out = block(hidden_channels, out_channels, 1, 1, 0, bias=biases[1], nonlinearity=nonlinearity, order=order[0:3], inplace_nonlinearity=inplace_nonlinearity, **residual_params)
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels, bias=biases[2], nonlinearity=skip_nonlinearity_type, order=order[0:3], stride=shortcut_stride, blur=shortcut_blur, **shortcut_params)
        elif in_channels < out_channels:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = skip_block(in_channels, out_channels - in_channels, bias=biases[2], nonlinearity=skip_nonlinearity_type, order=order[0:3], stride=shortcut_stride, blur=shortcut_blur, **shortcut_params)
        self.conditional = getattr(self.conv_block_0, 'conditional', False) or getattr(self.conv_block_1, 'conditional', False) or getattr(self.conv_block_1x1_in, 'conditional', False) or getattr(self.conv_block_1x1_out, 'conditional', False)

    def _get_stride_blur(self):
        if self.stride > 1:
            first_stride, second_stride = 1, self.stride
            first_blur, second_blur = False, self.blur
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                if self.border_free:
                    self.resample = nn.AvgPool2d(2)
                else:
                    self.resample = BlurDownsample()
            else:
                shortcut_stride = self.stride
                self.resample = nn.AvgPool2d(2)
        elif self.stride < 1:
            first_stride, second_stride = self.stride, 1
            first_blur, second_blur = self.blur, False
            shortcut_blur = False
            shortcut_stride = 1
            if self.blur:
                if self.border_free:
                    self.resample = nn.Upsample(scale_factor=2, mode='bilinear')
                else:
                    self.resample = BlurUpsample()
            else:
                shortcut_stride = self.stride
                self.resample = nn.Upsample(scale_factor=2)
        else:
            first_stride = second_stride = 1
            first_blur = second_blur = False
            shortcut_stride = 1
            shortcut_blur = False
            self.resample = None
        return first_stride, second_stride, shortcut_stride, first_blur, second_blur, shortcut_blur

    def conv_blocks(self, x, *cond_inputs, separate_cond=False, **kw_cond_inputs):
        if separate_cond:
            assert len(list(cond_inputs)) == 4
            dx = self.conv_block_1x1_in(x, cond_inputs[0], **kw_cond_inputs.get('kwargs_0', {}))
            dx = self.conv_block_0(dx, cond_inputs[1], **kw_cond_inputs.get('kwargs_1', {}))
            dx = self.conv_block_1(dx, cond_inputs[2], **kw_cond_inputs.get('kwargs_2', {}))
            dx = self.conv_block_1x1_out(dx, cond_inputs[3], **kw_cond_inputs.get('kwargs_3', {}))
        else:
            dx = self.conv_block_1x1_in(x, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_0(dx, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
            dx = self.conv_block_1x1_out(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)
        if self.resample_first and self.resample is not None:
            x = self.resample(x)
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        elif self.in_channels < self.out_channels:
            x_shortcut_pad = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
            x_shortcut = torch.cat((x, x_shortcut_pad), dim=1)
        elif self.in_channels > self.out_channels:
            x_shortcut = x[:, :self.out_channels, :, :]
        else:
            x_shortcut = x
        if not self.resample_first and self.resample is not None:
            x_shortcut = self.resample(x_shortcut)
        output = x_shortcut + dx
        return self.output_scale * output

    def extra_repr(self):
        s = 'output_scale={output_scale}'
        return s.format(**self.__dict__)


class DeepRes2dBlock(_BaseDeepResBlock):
    """Residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, activation_norm_type='none', activation_norm_params=None, skip_activation_norm=True, skip_nonlinearity=False, skip_weight_norm=True, nonlinearity='leakyrelu', inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=False, output_scale=1, blur=True, resample_first=True, border_free=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise, hidden_channels_equal_out_channels, order, Conv2dBlock, learn_shortcut, output_scale, blur=blur, resample_first=resample_first, border_free=border_free, skip_weight_norm=skip_weight_norm)


class ViT2dBlock(nn.Module):
    """An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, blur, order, input_dim, clamp, blur_kernel=(1, 3, 3, 1), output_scale=None, init_gain=1.0):
        super().__init__()
        self.weight_norm_type = weight_norm_type
        self.stride = stride
        self.clamp = clamp
        self.init_gain = init_gain
        if 'fused' in nonlinearity:
            lr_mul = getattr(weight_norm_params, 'lr_mul', 1)
            conv_before_nonlinearity = order.find('C') < order.find('A')
            if conv_before_nonlinearity:
                assert bias
                bias = False
            channel = out_channels if conv_before_nonlinearity else in_channels
            nonlinearity_layer = get_nonlinearity_layer(nonlinearity, inplace=inplace_nonlinearity, num_channels=channel, lr_mul=lr_mul)
        else:
            nonlinearity_layer = get_nonlinearity_layer(nonlinearity, inplace=inplace_nonlinearity)
        if apply_noise:
            order = order.replace('C', 'CG')
            noise_layer = ApplyNoise()
        else:
            noise_layer = None
        if blur:
            if stride == 2:
                p = len(blur_kernel) - 2 + (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2, p // 2
                padding = 0
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode)
                order = order.replace('C', 'BC')
            elif stride == 0.5:
                padding = 0
                p = len(blur_kernel) - 2 - (kernel_size - 1)
                pad0, pad1 = (p + 1) // 2 + 1, p // 2 + 1
                blur_layer = Blur(blur_kernel, pad=(pad0, pad1), padding_mode=padding_mode)
                order = order.replace('C', 'CB')
            elif stride == 1:
                blur_layer = nn.Identity()
            else:
                raise NotImplementedError
        else:
            blur_layer = nn.Identity()
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim))
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(norm_channels, activation_norm_type, input_dim, **vars(activation_norm_params))
        mappings = {'C': {'conv': conv_layer}, 'N': {'norm': activation_norm_layer}, 'A': {'nonlinearity': nonlinearity_layer}}
        mappings.update({'B': {'blur': blur_layer}})
        mappings.update({'G': {'noise': noise_layer}})
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])
        self.conditional = getattr(conv_layer, 'conditional', False) or getattr(activation_norm_layer, 'conditional', False)
        if output_scale is not None:
            self.output_scale = nn.Parameter(torch.tensor(output_scale))
        else:
            self.register_parameter('output_scale', None)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        """

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for key, layer in self.layers.items():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
            if self.clamp is not None and isinstance(layer, nn.Conv2d):
                x.clamp_(max=self.clamp)
            if key == 'conv':
                if self.output_scale is not None:
                    x = x * self.output_scale
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, input_dim):
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            if stride < 1:
                padding_mode = 'zeros'
                assert padding == 0
                layer_type = getattr(nn, f'ConvTranspose{input_dim}d')
                stride = round(1 / stride)
            else:
                layer_type = getattr(nn, f'Conv{input_dim}d')
            layer = layer_type(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        return layer

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and self.weight_norm_type != '':
                mod_str = mod_str[:-1] + ', weight_norm={}'.format(self.weight_norm_type) + ')'
            if name == 'conv' and getattr(layer, 'base_lr_mul', 1) != 1:
                mod_str = mod_str[:-1] + ', lr_mul={}'.format(layer.base_lr_mul) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ' + line) for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class WeightDemodulation(nn.Module):
    """Weight demodulation in
    "Analyzing and Improving the Image Quality of StyleGAN", Karras et al.

    Args:
        conv (torch.nn.Modules): Convolutional layer.
        cond_dims (int): The number of channels in the conditional input.
        eps (float, optional, default=1e-8): a value added to the
            denominator for numerical stability.
        adaptive_bias (bool, optional, default=False): If ``True``, adaptively
            predicts bias from the conditional input.
        demod (bool, optional, default=False): If ``True``, performs
            weight demodulation.
    """

    def __init__(self, conv, cond_dims, eps=1e-08, adaptive_bias=False, demod=True):
        super().__init__()
        self.conv = conv
        self.adaptive_bias = adaptive_bias
        if adaptive_bias:
            self.conv.register_parameter('bias', None)
            self.fc_beta = LinearBlock(cond_dims, self.conv.out_channels)
        self.fc_gamma = LinearBlock(cond_dims, self.conv.in_channels)
        self.eps = eps
        self.demod = demod
        self.conditional = True

    def forward(self, x, y, **_kwargs):
        """Weight demodulation forward"""
        b, c, h, w = x.size()
        self.conv.groups = b
        gamma = self.fc_gamma(y)
        gamma = gamma[:, None, :, None, None]
        weight = self.conv.weight[None, :, :, :, :] * gamma
        if self.demod:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.conv.out_channels, *ws)
        x = self.conv._conv_forward(x, weight)
        x = x.reshape(-1, self.conv.out_channels, h, w)
        if self.adaptive_bias:
            x += self.fc_beta(y)[:, :, None, None]
        return x


class DictLoss(nn.Module):

    def __init__(self, criterion='l1'):
        super(DictLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake, real):
        """Return the target vector for the l1/l2 loss computation.

        Args:
           fake (dict, list or tuple): Discriminator features of fake images.
           real (dict, list or tuple): Discriminator features of real images.
        Returns:
           loss (tensor): Loss value.
        """
        loss = 0
        if type(fake) == dict:
            for key in fake.keys():
                loss += self.criterion(fake[key], real[key].detach())
        elif type(fake) == list or type(fake) == tuple:
            for f, r in zip(fake, real):
                loss += self.criterion(f, r.detach())
        else:
            loss += self.criterion(fake, real.detach())
        return loss


class FeatureMatchingLoss(nn.Module):
    """Compute feature matching loss"""

    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        """Return the target vector for the binary cross entropy loss
        computation.

        Args:
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.

        Returns:
           (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss


class MaskedL1Loss(nn.Module):
    """Masked L1 loss constructor."""

    def __init__(self, normalize_over_valid=False):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.normalize_over_valid = normalize_over_valid

    def forward(self, input, target, mask):
        """Masked L1 loss computation.

        Args:
            input (tensor): Input tensor.
            target (tensor): Target tensor.
            mask (tensor): Mask to be applied to the output loss.

        Returns:
            (tensor): Loss value.
        """
        mask = mask.expand_as(input)
        loss = self.criterion(input * mask, target * mask)
        if self.normalize_over_valid:
            loss = loss * torch.numel(mask) / (torch.sum(mask) + 1e-06)
        return loss


def get_face_mask(densepose_map):
    """Obtain mask of faces.
    Args:
        densepose_map (3D or 4D tensor): DensePose map.
    Returns:
        mask (3D or 4D tensor): Face mask.
    """
    need_reshape = densepose_map.dim() == 4
    if need_reshape:
        bo, t, h, w = densepose_map.size()
        densepose_map = densepose_map.view(-1, h, w)
    b, h, w = densepose_map.size()
    part_map = (densepose_map / 2 + 0.5) * 24
    assert (part_map >= 0).all() and (part_map < 25).all()
    if densepose_map.is_cuda:
        mask = torch.ByteTensor(b, h, w).fill_(0)
    else:
        mask = torch.ByteTensor(b, h, w).fill_(0)
    for j in [23, 24]:
        mask = mask | ((part_map > j - 0.1) & (part_map < j + 0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, h, w)
    return mask.float()


def get_fg_mask(densepose_map, has_fg):
    """Obtain the foreground mask for pose sequences, which only includes
    the human. This is done by looking at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
        has_fg (bool): Whether data has foreground or not.
    Returns:
        mask (Nx1xHxW tensor): fg mask.
    """
    if type(densepose_map) == list:
        return [get_fg_mask(label, has_fg) for label in densepose_map]
    if not has_fg or densepose_map is None:
        return 1
    if densepose_map.dim() == 5:
        densepose_map = densepose_map[:, 0]
    mask = densepose_map[:, 2:3]
    mask = torch.nn.MaxPool2d(15, padding=7, stride=1)(mask)
    mask = (mask > -1).float()
    return mask


def get_part_mask(densepose_map):
    """Obtain mask of different body parts of humans. This is done by
    looking at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
    Returns:
        mask (NxKxHxW tensor): Body part mask, where K is the number of parts.
    """
    part_groups = [[0], [1, 2], [3, 4], [5, 6], [7, 9, 8, 10], [11, 13, 12, 14], [15, 17, 16, 18], [19, 21, 20, 22], [23, 24]]
    n_parts = len(part_groups)
    need_reshape = densepose_map.dim() == 4
    if need_reshape:
        bo, t, h, w = densepose_map.size()
        densepose_map = densepose_map.view(-1, h, w)
    b, h, w = densepose_map.size()
    part_map = (densepose_map / 2 + 0.5) * 24
    assert (part_map >= 0).all() and (part_map < 25).all()
    mask = torch.ByteTensor(b, n_parts, h, w).fill_(0)
    for i in range(n_parts):
        for j in part_groups[i]:
            mask[:, i] = mask[:, i] | ((part_map > j - 0.1) & (part_map < j + 0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, -1, h, w)
    return mask.float()


def pick_image(images, idx):
    """Pick the image among images according to idx.

    Args:
        images (B x N x C x H x W tensor or list of tensors) : N images.
        idx (B tensor) : indices to select.
    Returns:
        image (B x C x H x W) : Selected images.
    """
    if type(images) == list:
        return [pick_image(r, idx) for r in images]
    if idx is None:
        return images[:, 0]
    elif type(idx) == int:
        return images[:, idx]
    idx = idx.long().view(-1, 1, 1, 1, 1)
    image = images.gather(1, idx.expand_as(images)[:, 0:1])[:, 0]
    return image


class FlowLoss(nn.Module):
    """Flow loss constructor.

    Args:
        cfg (obj): Configuration.
    """

    def __init__(self, cfg):
        super(FlowLoss, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.criterion = nn.L1Loss()
        self.criterionMasked = MaskedL1Loss()
        flow_module = importlib.import_module(cfg.flow_network.type)
        self.flowNet = flow_module.FlowNet(pretrained=True)
        self.warp_ref = getattr(cfg.gen.flow, 'warp_ref', False)
        self.pose_cfg = pose_cfg = getattr(cfg.data, 'for_pose_dataset', None)
        self.for_pose_dataset = pose_cfg is not None
        self.has_fg = getattr(cfg.data, 'has_foreground', False)

    def forward(self, data, net_G_output, current_epoch):
        """Compute losses on the output flow and occlusion mask.

        Args:
            data (dict): Input data.
            net_G_output (dict): Generator output.
            current_epoch (int): Current training epoch number.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and the
                target image when using the flow to warp.
              - loss_mask (tensor): Loss for the occlusion mask.
        """
        tgt_label, tgt_image = data['label'], data['image']
        fake_image = net_G_output['fake_images']
        warped_images = net_G_output['warped_images']
        flow = net_G_output['fake_flow_maps']
        occ_mask = net_G_output['fake_occlusion_masks']
        if self.warp_ref:
            ref_labels, ref_images = data['ref_labels'], data['ref_images']
            ref_idx = net_G_output['ref_idx']
            ref_label, ref_image = pick_image([ref_labels, ref_images], ref_idx)
        else:
            ref_label = ref_image = None
        flow_gt_prev = flow_gt_ref = conf_gt_prev = conf_gt_ref = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self.warp_ref:
                if self.for_pose_dataset:
                    flow_gt_ref, conf_gt_ref = self.flowNet(tgt_label[:, :3], ref_label[:, :3])
                else:
                    flow_gt_ref, conf_gt_ref = self.flowNet(tgt_image, ref_image)
            if current_epoch >= self.cfg.single_frame_epoch and data['real_prev_image'] is not None:
                tgt_image_prev = data['real_prev_image']
                flow_gt_prev, conf_gt_prev = self.flowNet(tgt_image, tgt_image_prev)
        flow_gt = [flow_gt_ref, flow_gt_prev]
        flow_conf_gt = [conf_gt_ref, conf_gt_prev]
        fg_mask, ref_fg_mask = get_fg_mask([tgt_label, ref_label], self.has_fg)
        loss_flow_L1, loss_flow_warp, body_mask_diff = self.compute_flow_losses(flow, warped_images, tgt_image, flow_gt, flow_conf_gt, fg_mask, tgt_label, ref_label)
        loss_mask = self.compute_mask_losses(occ_mask, fake_image, warped_images, tgt_label, tgt_image, fg_mask, ref_fg_mask, body_mask_diff)
        return loss_flow_L1, loss_flow_warp, loss_mask

    def compute_flow_losses(self, flow, warped_images, tgt_image, flow_gt, flow_conf_gt, fg_mask, tgt_label, ref_label):
        """Compute losses on the generated flow maps.

        Args:
            flow (tensor or list of tensors): Generated flow maps.
                warped_images (tensor or list of tensors): Warped images using the
                flow maps.
            tgt_image (tensor): Target image for the warped image.
                flow_gt (tensor or list of tensors): Ground truth flow maps.
            flow_conf_gt (tensor or list of tensors): Confidence for the ground
                truth flow maps.
            fg_mask (tensor): Foreground mask for the target image.
            tgt_label (tensor): Target label map.
            ref_label (tensor): Reference label map.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and the
                target image when using the flow to warp.
              - body_mask_diff (tensor): Difference between warped body part map
                and target body part map. Used for pose dataset only.
        """
        loss_flow_L1 = torch.tensor(0.0, device=torch.device('cuda'))
        loss_flow_warp = torch.tensor(0.0, device=torch.device('cuda'))
        if isinstance(flow, list):
            for i in range(len(flow)):
                loss_flow_L1_i, loss_flow_warp_i = self.compute_flow_loss(flow[i], warped_images[i], tgt_image, flow_gt[i], flow_conf_gt[i], fg_mask)
                loss_flow_L1 += loss_flow_L1_i
                loss_flow_warp += loss_flow_warp_i
        else:
            loss_flow_L1, loss_flow_warp = self.compute_flow_loss(flow, warped_images, tgt_image, flow_gt[-1], flow_conf_gt[-1], fg_mask)
        body_mask_diff = None
        if self.warp_ref:
            if self.for_pose_dataset:
                body_mask = get_part_mask(tgt_label[:, 2])
                ref_body_mask = get_part_mask(ref_label[:, 2])
                warped_ref_body_mask = resample(ref_body_mask, flow[0])
                loss_flow_warp += self.criterion(warped_ref_body_mask, body_mask)
                body_mask_diff = torch.sum(abs(warped_ref_body_mask - body_mask), dim=1, keepdim=True)
            if self.has_fg:
                fg_mask, ref_fg_mask = get_fg_mask([tgt_label, ref_label], True)
                warped_ref_fg_mask = resample(ref_fg_mask, flow[0])
                loss_flow_warp += self.criterion(warped_ref_fg_mask, fg_mask)
        return loss_flow_L1, loss_flow_warp, body_mask_diff

    def compute_flow_loss(self, flow, warped_image, tgt_image, flow_gt, flow_conf_gt, fg_mask):
        """Compute losses on the generated flow map.

        Args:
            flow (tensor): Generated flow map.
            warped_image (tensor): Warped image using the flow map.
            tgt_image (tensor): Target image for the warped image.
            flow_gt (tensor): Ground truth flow map.
            flow_conf_gt (tensor): Confidence for the ground truth flow map.
            fg_mask (tensor): Foreground mask for the target image.
        Returns:
            (dict):
              - loss_flow_L1 (tensor): L1 loss compared to ground truth flow.
              - loss_flow_warp (tensor): L1 loss between the warped image and
              the target image when using the flow to warp.
        """
        loss_flow_L1 = torch.tensor(0.0, device=torch.device('cuda'))
        loss_flow_warp = torch.tensor(0.0, device=torch.device('cuda'))
        if flow is not None and flow_gt is not None:
            loss_flow_L1 = self.criterionMasked(flow, flow_gt, flow_conf_gt * fg_mask)
        if warped_image is not None:
            loss_flow_warp = self.criterion(warped_image, tgt_image)
        return loss_flow_L1, loss_flow_warp

    def compute_mask_losses(self, occ_mask, fake_image, warped_image, tgt_label, tgt_image, fg_mask, ref_fg_mask, body_mask_diff):
        """Compute losses on the generated occlusion masks.

        Args:
            occ_mask (tensor or list of tensors): Generated occlusion masks.
            fake_image (tensor): Generated image.
            warped_image (tensor or list of tensors): Warped images using the
                flow maps.
            tgt_label (tensor): Target label map.
            tgt_image (tensor): Target image for the warped image.
            fg_mask (tensor): Foreground mask for the target image.
            ref_fg_mask (tensor): Foreground mask for the reference image.
            body_mask_diff (tensor): Difference between warped body part map
            and target body part map. Used for pose dataset only.
        Returns:
            (tensor): Loss for the mask.
        """
        loss_mask = torch.tensor(0.0, device=torch.device('cuda'))
        if isinstance(occ_mask, list):
            for i in range(len(occ_mask)):
                loss_mask += self.compute_mask_loss(occ_mask[i], warped_image[i], tgt_image)
        else:
            loss_mask += self.compute_mask_loss(occ_mask, warped_image, tgt_image)
        if self.warp_ref:
            ref_occ_mask = occ_mask[0]
            dummy0 = torch.zeros_like(ref_occ_mask)
            dummy1 = torch.ones_like(ref_occ_mask)
            if self.for_pose_dataset:
                face_mask = get_face_mask(tgt_label[:, 2]).unsqueeze(1)
                AvgPool = torch.nn.AvgPool2d(15, padding=7, stride=1)
                face_mask = AvgPool(face_mask)
                loss_mask += self.criterionMasked(ref_occ_mask, dummy0, face_mask)
                loss_mask += self.criterionMasked(fake_image, warped_image[0], face_mask)
                loss_mask += self.criterionMasked(ref_occ_mask, dummy1, body_mask_diff)
            if self.has_fg:
                fg_mask_diff = (ref_fg_mask - fg_mask > 0).float()
                loss_mask += self.criterionMasked(ref_occ_mask, dummy1, fg_mask_diff)
        return loss_mask

    def compute_mask_loss(self, occ_mask, warped_image, tgt_image):
        """Compute losses on the generated occlusion mask.

        Args:
            occ_mask (tensor): Generated occlusion mask.
            warped_image (tensor): Warped image using the flow map.
            tgt_image (tensor): Target image for the warped image.
        Returns:
            (tensor): Loss for the mask.
        """
        loss_mask = torch.tensor(0.0, device=torch.device('cuda'))
        if occ_mask is not None:
            dummy0 = torch.zeros_like(occ_mask)
            dummy1 = torch.ones_like(occ_mask)
            img_diff = torch.sum(abs(warped_image - tgt_image), dim=1, keepdim=True)
            conf = torch.clamp(1 - img_diff, 0, 1)
            loss_mask = self.criterionMasked(occ_mask, dummy0, conf)
            loss_mask += self.criterionMasked(occ_mask, dummy1, 1 - conf)
        return loss_mask


class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """GAN loss constructor.

        Args:
            target_real_label (float): Desired output label for the real images.
            target_fake_label (float): Desired output label for the fake images.
        """
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None

    def forward(self, input_x, t_real, weight=None, reduce_dim=True, dis_update=True):
        """GAN loss computation.

        Args:
            input_x (tensor or list of tensors): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(input_x, list):
            loss = 0
            for pred_i in input_x:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, t_real, weight, reduce_dim, dis_update)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input_x)
        else:
            return self.loss(input_x, t_real, weight, reduce_dim, dis_update)

    def loss(self, input_x, t_real, weight=None, reduce_dim=True, dis_update=True):
        """N+1 label GAN loss computation.

        Args:
            input_x (tensor): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        assert reduce_dim is True
        pred = input_x['pred'].clone()
        label = input_x['label'].clone()
        batch_size = pred.size(0)
        label[:, 0, ...] = 0
        pred[:, 0, ...] = 0
        pred = F.log_softmax(pred, dim=1)
        assert pred.size(1) == label.size(1) + 1
        if dis_update:
            if t_real:
                pred_real = pred[:, :-1, :, :]
                loss = -label * pred_real
                loss = torch.sum(loss, dim=1, keepdim=True)
            else:
                pred_fake = pred[:, -1, None, :, :]
                loss = -pred_fake
        else:
            assert t_real, 'GAN loss must be aiming for real.'
            pred_real = pred[:, :-1, :, :]
            loss = -label * pred_real
            loss = torch.sum(loss, dim=1, keepdim=True)
        if weight is not None:
            loss = loss * weight
        if reduce_dim:
            loss = torch.mean(loss)
        else:
            loss = loss.view(batch_size, -1).mean(dim=1)
        return loss


def dist_all_reduce_tensor(tensor, reduce='mean'):
    """ Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        if reduce == 'mean':
            tensor /= world_size
        elif reduce == 'sum':
            pass
        else:
            raise NotImplementedError
    return tensor


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        all_grads = torch.stack(grads)
        all_grads = dist_all_reduce_tensor(all_grads, reduce='sum')
        grad_out[:] = all_grads[get_rank()]
        return grad_out


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07, gather_distributed=True, learn_temperature=True, single_direction=False, flatten=True):
        super(InfoNCELoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor([math.log(1 / temperature)]))
        self.logit_scale.requires_grad = learn_temperature
        self.gather_distributed = gather_distributed
        self.single_direction = single_direction
        self.flatten = flatten

    def forward(self, features_a, features_b, gather_distributed=None, eps=1e-08):
        if gather_distributed is None:
            gather_distributed = self.gather_distributed
        if features_a is None or features_b is None:
            return torch.tensor(0, device='cuda'), torch.tensor(0, device='cuda')
        bs_a, bs_b = features_a.size(0), features_b.size(0)
        if self.flatten:
            features_a, features_b = features_a.reshape(bs_a, -1), features_b.reshape(bs_b, -1)
        else:
            features_a = features_a.reshape(bs_a, features_a.size(1), -1).mean(-1)
            features_b = features_b.reshape(bs_b, features_b.size(1), -1).mean(-1)
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        features_a = features_a / (features_a.norm(dim=1, keepdim=True) + eps)
        features_b = features_b / (features_b.norm(dim=1, keepdim=True) + eps)
        loss_a = self._forward_single_direction(features_a, features_b, gather_distributed)
        if self.single_direction:
            return loss_a
        else:
            loss_b = self._forward_single_direction(features_b, features_a, gather_distributed)
            return loss_a + loss_b

    def _forward_single_direction(self, features_a, features_b, gather_distributed):
        bs_a = features_a.shape[0]
        logit_scale = self.logit_scale.exp()
        if get_world_size() > 1 and gather_distributed:
            gather_features_b = torch.cat(GatherLayer.apply(features_b))
            gather_labels_a = torch.arange(bs_a, device='cuda') + get_rank() * bs_a
            logits_a = logit_scale * features_a @ gather_features_b.t()
        else:
            gather_labels_a = torch.arange(bs_a, device='cuda')
            logits_a = logit_scale * features_a @ features_b.t()
        loss_a = F.cross_entropy(logits_a, gather_labels_a)
        return loss_a


class GaussianKLLoss(nn.Module):
    """Compute KL loss in VAE for Gaussian distributions"""

    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar=None):
        """Compute loss

        Args:
            mu (tensor): mean
            logvar (tensor): logarithm of variance
        """
        if logvar is None:
            logvar = torch.zeros_like(mu)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


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


def _alexnet(layers):
    """Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {(0): 'conv_1', (1): 'relu_1', (3): 'conv_2', (4): 'relu_2', (6): 'conv_3', (7): 'relu_3', (8): 'conv_4', (9): 'relu_4', (10): 'conv_5', (11): 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    """Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2), inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2), inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e, inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {(3): 'pool_1', (6): 'pool_2', (14): 'mixed_6e', (18): 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    """Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4, resnet50.avgpool)
    layer_name_mapping = {(4): 'layer_1', (5): 'layer_2', (6): 'layer_3', (7): 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    """Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url('http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4, resnet50.avgpool)
    layer_name_mapping = {(4): 'layer_1', (5): 'layer_2', (6): 'layer_3', (7): 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    """Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1', (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15): 'relu_3_3', (18): 'relu_4_1', (20): 'relu_4_2', (22): 'relu_4_3', (25): 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg19(layers):
    """Get vgg19 layers"""
    vgg = torchvision.models.vgg19(pretrained=True)
    network = torch.nn.Sequential(*(list(vgg.features) + [vgg.avgpool] + [nn.Flatten()] + list(vgg.classifier)))
    layer_name_mapping = {(1): 'relu_1_1', (3): 'relu_1_2', (6): 'relu_2_1', (8): 'relu_2_2', (11): 'relu_3_1', (13): 'relu_3_2', (15): 'relu_3_3', (17): 'relu_3_4', (20): 'relu_4_1', (22): 'relu_4_2', (24): 'relu_4_3', (26): 'relu_4_4', (29): 'relu_5_1', (31): 'relu_5_2', (33): 'relu_5_3', (35): 'relu_5_4', (36): 'pool_5', (42): 'fc_2'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url('http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth')
    feature_layer_name_mapping = {(0): 'conv1_1', (2): 'conv1_2', (5): 'conv2_1', (7): 'conv2_2', (10): 'conv3_1', (12): 'conv3_2', (14): 'conv3_3', (17): 'conv4_1', (19): 'conv4_2', (21): 'conv4_3', (24): 'conv5_1', (26): 'conv5_2', (28): 'conv5_3'}
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict['features.' + str(k) + '.weight'] = state_dict[v + '.weight']
        new_state_dict['features.' + str(k) + '.bias'] = state_dict[v + '.bias']
    classifier_layer_name_mapping = {(0): 'fc6', (3): 'fc7', (6): 'fc8'}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict['classifier.' + str(k) + '.weight'] = state_dict[v + '.weight']
        new_state_dict['classifier.' + str(k) + '.bias'] = state_dict[v + '.bias']
    network.load_state_dict(new_state_dict)


    class Flatten(nn.Module):

        def forward(self, x):
            return x.view(x.shape[0], -1)
    layer_name_mapping = {(0): 'conv_1_1', (1): 'relu_1_1', (2): 'conv_1_2', (5): 'conv_2_1', (6): 'relu_2_1', (7): 'conv_2_2', (10): 'conv_3_1', (11): 'relu_3_1', (12): 'conv_3_2', (14): 'conv_3_3', (17): 'conv_4_1', (18): 'relu_4_1', (19): 'conv_4_2', (21): 'conv_4_3', (24): 'conv_5_1', (25): 'relu_5_1', (26): 'conv_5_2', (28): 'conv_5_3', (33): 'fc6', (36): 'fc7', (39): 'fc8'}
    seq_layers = []
    for feature in network.features:
        seq_layers += [feature]
    seq_layers += [network.avgpool, Flatten()]
    for classifier in network.classifier:
        seq_layers += [classifier]
    network = nn.Sequential(*seq_layers)
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def apply_imagenet_normalization(input):
    """Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    normalized_input = (input + 1) / 2
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


def is_local_master():
    return torch.cuda.current_device() == 0


string_classes = str, bytes


def to_float(data):
    """Move all halfs to float.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.float()
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_float(data[key]) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [to_float(d) for d in data]
    else:
        return data


class PerceptualLoss(nn.Module):
    """Perceptual loss initialization.

    Args:
       network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
       layers (str or list of str) : The layers used to compute the loss.
       weights (float or list of float : The loss weights of each layer.
       criterion (str): The type of distance function: 'l1' | 'l2'.
       resize (bool) : If ``True``, resize the input images to 224x224.
       resize_mode (str): Algorithm used for resizing.
       num_scales (int): The loss will be evaluated at original size and
        this many times downsampled sizes.
       per_sample_weight (bool): Output loss for individual samples in the
        batch instead of mean loss.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None, criterion='l1', resize=False, resize_mode='bilinear', num_scales=1, per_sample_weight=False, info_nce_temperature=0.07, info_nce_gather_distributed=True, info_nce_learn_temperature=True, info_nce_flatten=True):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.0] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]
        if dist.is_initialized() and not is_local_master():
            torch.distributed.barrier()
        assert len(layers) == len(weights), 'The number of layers (%s) must be equal to the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)
        if dist.is_initialized() and is_local_master():
            torch.distributed.barrier()
        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        reduction = 'mean' if not per_sample_weight else 'none'
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif criterion == 'info_nce':
            self.criterion = InfoNCELoss(temperature=info_nce_temperature, gather_distributed=info_nce_gather_distributed, learn_temperature=info_nce_learn_temperature, flatten=info_nce_flatten, single_direction=True)
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        None
        None

    def forward(self, inp, target, per_sample_weights=None):
        """Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.
           per_sample_weight (bool): Output loss for individual samples in the
            batch instead of mean loss.
        Returns:
           (scalar tensor) : The perceptual loss.
        """
        if not torch.is_autocast_enabled():
            inp, target = to_float([inp, target])
        self.model.eval()
        inp, target = apply_imagenet_normalization(inp), apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(inp, mode=self.resize_mode, size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode=self.resize_mode, size=(224, 224), align_corners=False)
        loss = 0
        for scale in range(self.num_scales):
            input_features, target_features = self.model(inp), self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                l_tmp = self.criterion(input_features[layer], target_features[layer].detach())
                if per_sample_weights is not None:
                    l_tmp = l_tmp.mean(1).mean(1).mean(1)
                loss += weight * l_tmp
            if scale != self.num_scales - 1:
                inp = F.interpolate(inp, mode=self.resize_mode, scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(target, mode=self.resize_mode, scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
        return loss.float()


class WeightedMSELoss(nn.Module):
    """Compute Weighted MSE loss"""

    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        """Return weighted MSE Loss.
        Args:
           input (tensor):
           target (tensor):
           weight (tensor):
        Returns:
           (tensor): Loss value.
        """
        if self.reduction == 'mean':
            loss = torch.mean(weight * (input - target) ** 2)
        else:
            loss = torch.sum(weight * (input - target) ** 2)
        return loss


def calc_height_map(voxel_t):
    """Calculate height map given a voxel grid [Y, X, Z] as input.
    The height is defined as the Y index of the surface (non-air) block

    Args:
        voxel (Y x X x Z torch.IntTensor, CPU): Input voxel of three dimensions
    Output:
        heightmap (X x Z torch.IntTensor)
    """
    start_time = time.time()
    m, h = torch.max((torch.flip(voxel_t, [0]) != 0).int(), dim=0, keepdim=False)
    heightmap = voxel_t.shape[0] - 1 - h
    heightmap[m == 0] = 0
    elapsed_time = time.time() - start_time
    None
    return heightmap


def gen_corner_voxel(voxel):
    """Converting voxel center array to voxel corner array. The size of the
    produced array grows by 1 on every dimension.

    Args:
        voxel (torch.IntTensor, CPU): Input voxel of three dimensions
    """
    structure = np.zeros([3, 3, 3], dtype=np.bool)
    structure[1:, 1:, 1:] = True
    voxel_p = F.pad(voxel, (0, 1, 0, 1, 0, 1))
    corners = ndimage.binary_dilation(voxel_p.numpy(), structure)
    corners = torch.tensor(corners, dtype=torch.int32)
    return corners


def trans_vec_homo(m, v, is_vec=False):
    """3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    if is_vec:
        v = torch.tensor([v[0], v[1], v[2], 0], dtype=v.dtype)
    else:
        v = torch.tensor([v[0], v[1], v[2], 1], dtype=v.dtype)
    v = torch.mv(m, v)
    if not is_vec:
        v = v / v[3]
    v = v[:3]
    return v


class McVoxel(nn.Module):
    """Voxel management."""

    def __init__(self, voxel_t, preproc_ver):
        super(McVoxel, self).__init__()
        voxel_t[voxel_t == 246] = 0
        voxel_t[voxel_t == 241] = 0
        voxel_t[voxel_t == 611] = 26
        voxel_t[voxel_t == 183] = 26
        voxel_t[voxel_t == 401] = 25
        if preproc_ver >= 3 and preproc_ver < 6:
            voxel_t[voxel_t == 27] = 25
            voxel_t[voxel_t == 616] = 9
            voxel_t[voxel_t == 617] = 25
        if preproc_ver >= 6:
            voxel_t[voxel_t == 616] = 0
            voxel_t[voxel_t == 617] = 0
        structure = ndimage.generate_binary_structure(3, 3)
        mask = voxel_t.numpy() > 0
        if preproc_ver == 4:
            mask = ndimage.morphology.binary_erosion(mask, structure=structure, iterations=2, border_value=1)
            voxel_t[mask] = 0
        if preproc_ver >= 5:
            mask = ndimage.morphology.binary_dilation(mask, iterations=1, border_value=1)
            mask = ndimage.morphology.binary_erosion(mask, iterations=1, border_value=1)
            mask = ndimage.morphology.binary_erosion(mask, structure=structure, iterations=2, border_value=1)
            voxel_t[mask] = 0
        self.register_buffer('voxel_t', voxel_t, persistent=False)
        self.trans_mat = torch.eye(4)
        self.heightmap = calc_height_map(self.voxel_t)
        self._truncate_voxel()
        corner_t = gen_corner_voxel(self.voxel_t)
        self.register_buffer('corner_t', corner_t, persistent=False)
        nfilledvox = torch.sum(self.corner_t > 0)
        None
        self.corner_t[self.corner_t > 0] = torch.arange(start=1, end=nfilledvox + 1, step=1, dtype=torch.int32)
        self.nfilledvox = nfilledvox

    def world2local(self, v, is_vec=False):
        mat_world2local = torch.inverse(self.trans_mat)
        return trans_vec_homo(mat_world2local, v, is_vec)

    def _truncate_voxel(self):
        gnd_level = self.heightmap.min()
        sky_level = self.heightmap.max() + 1
        self.voxel_t = self.voxel_t[gnd_level:sky_level, :, :]
        self.trans_mat[0, 3] += gnd_level
        None

    def is_sea(self, loc):
        """loc: [2]: x, z."""
        x = int(loc[1])
        z = int(loc[2])
        if x < 0 or x > self.heightmap.size(0) or z < 0 or z > self.heightmap.size(1):
            None
            return True
        y = self.heightmap[x, z] - self.trans_mat[0, 3]
        y = int(y)
        if self.voxel_t[y, x, z] == 26:
            None
            None
            return True
        else:
            return False


_null_tensor = torch.empty([0])


activation_funcs = {'linear': SimpleNamespace(func=lambda x, **_: x, def_alpha=0, def_gain=1, cuda_idx=1, ref='', has_2nd_grad=False), 'relu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.relu(x), def_alpha=0, def_gain=np.sqrt(2), cuda_idx=2, ref='y', has_2nd_grad=False), 'leakyrelu': SimpleNamespace(func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha), def_alpha=0.2, def_gain=np.sqrt(2), cuda_idx=3, ref='y', has_2nd_grad=False), 'tanh': SimpleNamespace(func=lambda x, **_: torch.tanh(x), def_alpha=0, def_gain=1, cuda_idx=4, ref='y', has_2nd_grad=True), 'sigmoid': SimpleNamespace(func=lambda x, **_: torch.sigmoid(x), def_alpha=0, def_gain=1, cuda_idx=5, ref='y', has_2nd_grad=True), 'elu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.elu(x), def_alpha=0, def_gain=1, cuda_idx=6, ref='y', has_2nd_grad=True), 'selu': SimpleNamespace(func=lambda x, **_: torch.nn.functional.selu(x), def_alpha=0, def_gain=1, cuda_idx=7, ref='y', has_2nd_grad=True), 'softplus': SimpleNamespace(func=lambda x, **_: torch.nn.functional.softplus(x), def_alpha=0, def_gain=1, cuda_idx=8, ref='y', has_2nd_grad=True), 'swish': SimpleNamespace(func=lambda x, **_: torch.sigmoid(x) * x, def_alpha=0, def_gain=np.sqrt(2), cuda_idx=9, ref='x', has_2nd_grad=True)}


def _bias_act_cuda(dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Fast CUDA implementation of `bias_act()` using custom ops.
    """
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)
    key = dim, act, alpha, gain, clamp
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]


    class BiasActCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, b):
            if x.ndim > 2 and x.stride()[1] == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not _null_tensor:
                y = bias_act_cuda.bias_act_cuda(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor, b if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor, y if 'y' in spec.ref else _null_tensor)
            return y

        @staticmethod
        def backward(ctx, dy):
            dy = dy.contiguous(memory_format=ctx.memory_format)
            x, b, y = ctx.saved_tensors
            dx = None
            db = None
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    dx = BiasActCudaGrad.apply(dy, x, b, y)
            if ctx.needs_input_grad[1]:
                db = dx.sum([i for i in range(dx.ndim) if i != dim])
            return dx, db


    class BiasActCudaGrad(torch.autograd.Function):

        @staticmethod
        def forward(ctx, dy, x, b, y):
            if x.ndim > 2 and x.stride()[1] == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format
            dx = bias_act_cuda.bias_act_cuda(dy, b, x, y, _null_tensor, 1, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(dy if spec.has_2nd_grad else _null_tensor, x, b, y)
            return dx

        @staticmethod
        def backward(ctx, d_dx):
            d_dx = d_dx.contiguous(memory_format=ctx.memory_format)
            dy, x, b, y = ctx.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None
            if ctx.needs_input_grad[0]:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)
            if spec.has_2nd_grad and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                d_x = bias_act_cuda.bias_act_cuda(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx, alpha, gain, clamp)
            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])
            return d_dy, d_x, d_b, d_y
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda


def _bias_act_ref(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([(-1 if i == dim else 1) for i in range(x.ndim)])
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)
    gain = float(gain)
    if gain != 1:
        x = x * gain
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)
    return x


def _bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda':
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)


class FusedNonlinearity(nn.Module):

    def __init__(self, nonlinearity, num_channels=None, lr_mul=1.0, alpha=None, impl='cuda', gain=None):
        super().__init__()
        if num_channels is not None:
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('bias', None)
        self.nonlinearity = nonlinearity
        self.gain = gain
        self.alpha = alpha
        self.lr_mul = lr_mul
        self.impl = impl

    def forward(self, x):
        bias = self.bias.type_as(x) * self.lr_mul if self.bias is not None else None
        return _bias_act(x, b=bias, dim=1, act=self.nonlinearity, alpha=self.alpha, gain=self.gain, clamp=None, impl=self.impl)

    def __repr__(self):
        mod_str = f'{self.__class__.__name__}(type={self.nonlinearity}'
        if self.gain is not None:
            mod_str += f', gain={self.gain}'
        if self.alpha is not None:
            mod_str += f', alpha={self.alpha}'
        if self.lr_mul != 1:
            mod_str += f', lr_mul={self.lr_mul}'
        mod_str += ')'
        return mod_str


class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()
        channelnorm_cuda.forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)
        ctx.norm_deg = norm_deg
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        channelnorm_cuda.backward(input1, output, grad_output.data, grad_input1.data, ctx.norm_deg)
        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply, input1, input2):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction.apply(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply, input1, input2)
        return result


def get_confirm_token(response):
    """Get confirm token

    Args:
        response: Check if the file exists.

    Returns:

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    """Save response content

    Args:
        response:
        destination: Path to save the file.

    Returns:

    """
    chunk_size = 32768
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_file(URL, destination):
    """Download a file from google drive or pbss by using the url.

    Args:
        URL: GDrive URL or PBSS pre-signed URL for the checkpoint.
        destination: Path to save the file.

    Returns:

    """
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def is_master():
    """check if current process is the master"""
    return get_rank() == 0


def get_checkpoint(checkpoint_path, url=''):
    """Get the checkpoint path. If it does not exist yet, download it from
    the url.

    Args:
        checkpoint_path (str): Checkpoint path.
        url (str): URL to download checkpoint.
    Returns:
        (str): Full checkpoint path.
    """
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.getcwd()
    save_dir = os.path.join(os.environ['TORCH_HOME'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        if is_master():
            None
            if 'pbss.s8k.io' not in url:
                url = f'https://docs.google.com/uc?export=download&id={url}'
            download_file(url, full_checkpoint_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return full_checkpoint_path


class FlowNet(nn.Module):

    def __init__(self, pretrained=True, fp16=False):
        super().__init__()
        flownet2_args = types.SimpleNamespace()
        setattr(flownet2_args, 'fp16', fp16)
        setattr(flownet2_args, 'rgb_max', 1.0)
        if fp16:
            None
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2'](flownet2_args)
        if pretrained:
            flownet2_path = get_checkpoint('flownet2.pth.tar', '1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da')
            checkpoint = torch.load(flownet2_path, map_location=torch.device('cpu'))
            self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval()

    def forward(self, input_A, input_B):
        size = input_A.size()
        assert len(size) == 4 or len(size) == 5 or len(size) == 6
        if len(size) >= 5:
            if len(size) == 5:
                b, n, c, h, w = size
            else:
                b, t, n, c, h, w = size
            input_A = input_A.contiguous().view(-1, c, h, w)
            input_B = input_B.contiguous().view(-1, c, h, w)
            flow, conf = self.compute_flow_and_conf(input_A, input_B)
            if len(size) == 5:
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return flow.view(b, t, n, 2, h, w), conf.view(b, t, n, 1, h, w)
        else:
            return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert im1.size()[1] == 3
        assert im1.size() == im2.size()
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h // 64 * 64, old_w // 64 * 64
        if old_h != new_h:
            im1 = F.interpolate(im1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            im2 = F.interpolate(im2, size=(new_h, new_w), mode='bilinear', align_corners=False)
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        with torch.no_grad():
            flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - resample(im2, flow1)) < 0.02).float()
        if old_h != new_h:
            flow1 = F.interpolate(flow1, size=(old_h, old_w), mode='bilinear', align_corners=False) * old_h / new_h
            conf = F.interpolate(conf, size=(old_h, old_w), mode='bilinear', align_corners=False)
        return flow1, conf

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class FlowNet2(nn.Module):

    def __init__(self, args, use_batch_norm=False, div_flow=20.0):
        super(FlowNet2, self).__init__()
        self.batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        self.flownetc = flownet_c.FlowNetC(args, use_batch_norm=self.batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.args = args
        self.resample1 = resample2d.Resample2d()
        self.flownets_1 = flownet_s.FlowNetS(args, use_batch_norm=self.batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.resample2 = resample2d.Resample2d()
        self.flownets_2 = flownet_s.FlowNetS(args, use_batch_norm=self.batch_norm)
        self.flownets_d = flownet_sd.FlowNetSD(args, use_batch_norm=self.batch_norm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.resample3 = resample2d.Resample2d()
        self.resample4 = resample2d.Resample2d()
        self.flownetfusion = flownet_fusion.FlowNetFusion(args, use_batch_norm=self.batch_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        height, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([height, width])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.0)
        for i in range(min_dim):
            weight.data[i, i, :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]), flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        if self.args.fp16:
            resampled_img1 = self.resample2(tofp32()(x[:, 3:, :, :]), flownets1_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        if self.args.fp16:
            diff_flownets2_flow = self.resample4(tofp32()(x[:, 3:, :, :]), flownets2_flow)
            diff_flownets2_flow = tofp16()(diff_flownets2_flow)
        else:
            diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownets2_flow)
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        if self.args.fp16:
            diff_flownetsd_flow = self.resample3(tofp32()(x[:, 3:, :, :]), flownetsd_flow)
            diff_flownetsd_flow = tofp16()(diff_flownetsd_flow)
        else:
            diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownetsd_flow)
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        return flownetfusion_flow


class FlowNet2CS(nn.Module):

    def __init__(self, args, use_batch_norm=False, div_flow=20.0):
        super(FlowNet2CS, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        self.flownetc = flownet_c.FlowNetC(args, use_batch_norm=self.use_batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.args = args
        self.resample1 = resample2d.Resample2d()
        self.flownets_1 = flownet_s.FlowNetS(args, use_batch_norm=self.use_batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]), flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, args, use_batch_norm=False, div_flow=20.0):
        super(FlowNet2CSS, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = channelnorm.ChannelNorm()
        self.flownetc = flownet_c.FlowNetC(args, use_batch_norm=self.use_batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.args = args
        self.resample1 = resample2d.Resample2d()
        self.flownets_1 = flownet_s.FlowNetS(args, use_batch_norm=self.use_batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.resample2 = resample2d.Resample2d()
        self.flownets_2 = flownet_s.FlowNetS(args, use_batch_norm=self.use_batch_norm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest', align_corners=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        if self.args.fp16:
            resampled_img1 = self.resample1(tofp32()(x[:, 3:, :, :]), flownetc_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        if self.args.fp16:
            resampled_img1 = self.resample2(tofp32()(x[:, 3:, :, :]), flownets1_flow)
            resampled_img1 = tofp16()(resampled_img1)
        else:
            resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


def conv(use_batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True), nn.LeakyReLU(0.1, inplace=True))


def deconv(in_planes, out_planes):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class FlowNetC(nn.Module):

    def __init__(self, args, use_batch_norm=True, div_flow=20):
        """FlowNet2 C module. Check out the FlowNet2 paper for more details
        https://arxiv.org/abs/1612.01925

        Args:
            args (obj): Network initialization arguments
            use_batch_norm (bool): Use batch norm or not. Default is true.
            div_flow (int): Flow devision factor. Default is 20.
        """
        super(FlowNetC, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.div_flow = div_flow
        self.conv1 = conv(self.use_batch_norm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.use_batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.use_batch_norm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.use_batch_norm, 256, 32, kernel_size=1, stride=1)
        self.args = args
        self.corr = correlation.Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.use_batch_norm, 473, 256)
        self.conv4 = conv(self.use_batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.use_batch_norm, 512, 512)
        self.conv5 = conv(self.use_batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.use_batch_norm, 512, 512)
        self.conv6 = conv(self.use_batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.use_batch_norm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:, :, :]
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        if self.args.fp16:
            out_corr = self.corr(tofp32()(out_conv3a), tofp32()(out_conv3b))
            out_corr = tofp16()(out_corr)
        else:
            out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


def i_conv(use_batch_norm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias), nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias))


class FlowNetFusion(nn.Module):
    """FlowNet2 Fusion module. Check out the FlowNet2 paper for more details
    https://arxiv.org/abs/1612.01925

    Args:
        args (obj): Network initialization arguments
        use_batch_norm (bool): Use batch norm or not. Default is true.
    """

    def __init__(self, args, use_batch_norm=True):
        super(FlowNetFusion, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv0 = conv(self.use_batch_norm, 11, 64)
        self.conv1 = conv(self.use_batch_norm, 64, 64, stride=2)
        self.conv1_1 = conv(self.use_batch_norm, 64, 128)
        self.conv2 = conv(self.use_batch_norm, 128, 128, stride=2)
        self.conv2_1 = conv(self.use_batch_norm, 128, 128)
        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)
        self.inter_conv1 = i_conv(self.use_batch_norm, 162, 32)
        self.inter_conv0 = i_conv(self.use_batch_norm, 82, 16)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)
        return flow0


class FlowNetS(nn.Module):
    """FlowNet2 S module. Check out the FlowNet2 paper for more details
    https://arxiv.org/abs/1612.01925

    Args:
        args (obj): Network initialization arguments
        input_channels (int): Number of input channels. Default is 12.
        use_batch_norm (bool): Use batch norm or not. Default is true.
    """

    def __init__(self, args, input_channels=12, use_batch_norm=True):
        super(FlowNetS, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = conv(self.use_batch_norm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.use_batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.use_batch_norm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.use_batch_norm, 256, 256)
        self.conv4 = conv(self.use_batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.use_batch_norm, 512, 512)
        self.conv5 = conv(self.use_batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.use_batch_norm, 512, 512)
        self.conv6 = conv(self.use_batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.use_batch_norm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetSD(nn.Module):
    """FlowNet2 SD module. Check out the FlowNet2 paper for more details
    https://arxiv.org/abs/1612.01925

    Args:
        args (obj): Network initialization arguments
        use_batch_norm (bool): Use batch norm or not. Default is true.
    """

    def __init__(self, args, use_batch_norm=True):
        super(FlowNetSD, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv0 = conv(self.use_batch_norm, 6, 64)
        self.conv1 = conv(self.use_batch_norm, 64, 64, stride=2)
        self.conv1_1 = conv(self.use_batch_norm, 64, 128)
        self.conv2 = conv(self.use_batch_norm, 128, 128, stride=2)
        self.conv2_1 = conv(self.use_batch_norm, 128, 128)
        self.conv3 = conv(self.use_batch_norm, 128, 256, stride=2)
        self.conv3_1 = conv(self.use_batch_norm, 256, 256)
        self.conv4 = conv(self.use_batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.use_batch_norm, 512, 512)
        self.conv5 = conv(self.use_batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.use_batch_norm, 512, 512)
        self.conv6 = conv(self.use_batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.use_batch_norm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.inter_conv5 = i_conv(self.use_batch_norm, 1026, 512)
        self.inter_conv4 = i_conv(self.use_batch_norm, 770, 256)
        self.inter_conv3 = i_conv(self.use_batch_norm, 386, 128)
        self.inter_conv2 = i_conv(self.use_batch_norm, 194, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        """

        Args:
            x (tensor): Input tensors of concatenated images.
        Returns:
            flow2 (tensor): Output flow tensors.
        """
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = True
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        resample2d_cuda.forward(input1, input2, output, kernel_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        resample2d_cuda.backward(input1, input2, grad_output.data, grad_input1.data, grad_input2.data, ctx.kernel_size)
        return grad_input1, grad_input2, None, None


class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    @autocast(False)
    def forward(self, input1, input2):
        input1, input2 = input1.float(), input2.float()
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineMod,
     lambda: ([], {'in_features': 4, 'style_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 4])], {}),
     False),
    (ApplyNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Blur,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BlurDownsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BlurUpsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConstantInput,
     lambda: ([], {'channel': 4}),
     lambda: ([], {}),
     True),
    (DictLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Embedding2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
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
    (FeatureMatchingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FlowNetFusion,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 11, 64, 64])], {}),
     True),
    (FlowNetS,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (FlowNetSD,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     False),
    (GaussianKLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HyperConv2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'style_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 4])], {}),
     False),
    (ModelAverage,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (NetLinLayer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PartialConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PartialSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SKYMLP,
     lambda: ([], {'in_channels': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (SplitMeanStd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyleMLP,
     lambda: ([], {'style_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwAV,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Unet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (WeightedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WrappedModel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp32,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (unetConv2,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_NVlabs_imaginaire(_paritybench_base):
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

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

