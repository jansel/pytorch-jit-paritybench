import sys
_module = sys.modules[__name__]
del sys
compare = _module
models_jittor = _module
as_mlp = _module
conv_mixer = _module
conv_mlp = _module
cycle_mlp = _module
dyna_mlp = _module
einops_my = _module
_backends = _module
_jittor_specific = _module
_torch_specific = _module
einops = _module
experimental = _module
indexing = _module
layers = _module
_einmix = _module
chainer = _module
gluon = _module
jittor = _module
keras = _module
oneflow = _module
tensorflow = _module
parsing = _module
g_mlp = _module
hire_mlp = _module
mlp_mixer = _module
morph_mlp = _module
ms_mlp = _module
raft_mlp = _module
repmlpnet = _module
res_mlp = _module
s2_mlp_v1 = _module
s2_mlp_v2 = _module
sequencer = _module
sparse_mlp = _module
swin_mlp = _module
utils = _module
dcn_v2 = _module
init = _module
tools = _module
vip = _module
wave_mlp = _module
models_pytorch = _module
active_mlp = _module
as_mlp = _module
conv_mixer = _module
conv_mlp = _module
cycle_mlp = _module
dyna_mlp = _module
g_mlp = _module
gfnet = _module
hire_mlp = _module
mlp_mixer = _module
morph_mlp = _module
ms_mlp = _module
raft_mlp = _module
repmlpnet = _module
res_mlp = _module
s2_mlp_v1 = _module
s2_mlp_v2 = _module
sequencer = _module
sparse_mlp = _module
swin_mlp = _module
shift_cuda = _module
vip = _module
wave_mlp = _module

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


import time


import numpy as np


from torch.hub import load_state_dict_from_url


import math


import collections


from itertools import repeat


import warnings


from typing import Dict


from typing import List


import functools


import itertools


import typing


from collections import OrderedDict


from typing import Tuple


from typing import Union


from typing import Callable


from typing import Optional


from typing import TypeVar


from functools import partial


from functools import reduce


from abc import ABC


import torch.nn as nn


from torch.nn import init


from torch.nn.modules.utils import _pair


from torchvision.ops.deform_conv import deform_conv2d


import torch.nn.functional as F


import torch.utils.checkpoint as checkpoint


from torch import Tensor


from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv


from torch import nn


from torch.nn import functional as F


import logging


from numpy.lib.arraypad import pad


import torch.fft


from torch.autograd import Function


from collections import namedtuple


from string import Template


class DropPath(nn.Module):

    def __init__(self, drop_path_rate=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_path_rate

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random.floor_()
        return x.div(keep_prob) * random


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


CUDA_NUM_THREADS = 1024


def Dtype(t):
    if isinstance(t, torch.FloatTensor):
        return 'float'
    elif isinstance(t, torch.DoubleTensor):
        return 'double'


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


Stream = namedtuple('Stream', ['ptr'])


kernel_loop = """
#define CUDA_KERNEL_LOOP(i, n)                          for (int i = blockIdx.x * blockDim.x + threadIdx.x;       i < (n);                                             i += blockDim.x * gridDim.x)
"""


_shift_kernel = kernel_loop + """
extern "C"
__global__ void shift_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));
    ${Dtype} value = 0;
    if (${dim} == 2){
        if ((h + s >= 0 && h + s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h + s) * ${width} + w;
             value = bottom_data[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w + s >= 0 && w + s< ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w + s;
            value = bottom_data[offset];
            }
    }
    top_data[index] = value;
  }
}
"""


_shift_kernel_backward_grad_input = kernel_loop + """
extern "C"
__global__ void shift_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));
    ${Dtype} value = 0;
    if (${dim} == 2){
        if ((h - s >= 0 && h - s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h - s) * ${width} + w;
             value = top_diff[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w - s >= 0 && w - s < ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w - s;
            value = top_diff[offset];
        }
    }
    bottom_diff[index] = value;
  }
}
"""


class _shift(Function):

    @staticmethod
    def forward(ctx, input, shift, dim):
        batch_size, channels, height, width = input.size()
        output = input.new(batch_size, channels, height, width)
        n = output.numel()
        with torch.cuda.device_of(input):
            f = load_kernel('shift_forward_kernel', _shift_kernel, Dtype=Dtype(input), nthreads=n, num=batch_size, channels=channels, height=height, width=width, shift=shift, dim=dim, group=int(math.ceil(channels / shift)))
            f(block=(CUDA_NUM_THREADS, 1, 1), grid=(GET_BLOCKS(n), 1, 1), args=[input.data_ptr(), output.data_ptr()], stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input)
        ctx.shift, ctx.dim = shift, dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input = ctx.saved_tensors[0]
        shift, dim = ctx.shift, ctx.dim
        batch_size, channels, height, width = input.size()
        grad_input = None
        opt = dict(Dtype=Dtype(grad_output), num=batch_size, channels=channels, height=height, width=width, shift=shift, dim=dim, group=int(math.ceil(channels / shift)))
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('shift_backward_grad_input_kernel', _shift_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1), grid=(GET_BLOCKS(n), 1, 1), args=[grad_output.data_ptr(), grad_input.data_ptr()], stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, None, None


def _shift_cuda(input, shift, dim):
    """ involution kernel
    """
    assert shift >= 3 and shift % 2 == 1
    assert dim == 2 or dim == 3
    if input.is_cuda:
        out = _shift.apply(input, shift, dim)
    else:
        raise NotImplementedError
    return out


class Shift(nn.Module):

    def __init__(self, kernel_size, dim):
        super(Shift, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        assert dim == 2 or dim == 3
        assert kernel_size % 2 == 1

    def forward(self, x):
        if self.kernel_size == 1:
            return x
        out = _shift_cuda(x, self.kernel_size, self.dim)
        return out


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class AxialShift(nn.Module):
    """ Axial shift  
    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.actn = nn.GELU()
        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.shift_dim2 = Shift(self.shift_size, 2)
        self.shift_dim3 = Shift(self.shift_size, 3)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, C, H, W = x.shape
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
        """
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        
        xs = torch.chunk(x, self.shift_size, 1)
        def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat
        x_shift_lr = shift(3)
        x_shift_td = shift(2)
        """
        x_shift_lr = self.shift_dim3(x)
        x_shift_td = self.shift_dim2(x)
        x_lr = self.conv2_1(x_shift_lr)
        x_td = self.conv2_2(x_shift_td)
        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)
        x = x_lr + x_td
        x = self.norm2(x)
        x = self.conv3(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * self.dim
        flops += N * self.dim
        flops += N * self.dim * self.dim * 2
        flops += N * self.dim
        flops += N * self.dim
        flops += N * self.dim * self.dim
        return flops


class AxialShiftedBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=7, mlp_ratio=4.0, as_bias=True, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.axial_shift(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        flops += self.axial_shift.flops(H * W)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    """ Swin MLP Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.padding = [self.window_size - self.shift_size, self.shift_size, self.window_size - self.shift_size, self.shift_size]
        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2, self.num_heads * self.window_size ** 2, kernel_size=1, groups=self.num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], 'constant', 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        if self.shift_size > 0:
            nW = (H / self.window_size + 1) * (W / self.window_size + 1)
        else:
            nW = H * W / self.window_size / self.window_size
        flops += nW * self.dim * (self.window_size * self.window_size) * (self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinMLPBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def _no_grad_trunc_normal_(var, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    var.uniform_(low=2 * l - 1, high=2 * u - 1)
    var = var.erfinv()
    var = var.multiply(std * math.sqrt(2.0))
    var = var.add(mean)
    var = var.clamp(min_v=a, max_v=b)
    return var


def trunc_normal_(var, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input jt.jittor_core.Var with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        var: an n-dimensional `jt.jittor_core.Var` 
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(var, mean, std, a, b)


class AS_MLP(nn.Module):
    """ AS-MLP
        A PyTorch impl of : `AS-MLP: An Axial Shifted MLP Architecture for Vision`  -
          https://arxiv.org/pdf/xxx.xxx
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each AS-MLP layer.
        window_size (int): shift size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        as_bias (bool): If True, add a learnable bias to as-mlp block. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.GroupNorm with group=1.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], shift_size=5, mlp_ratio=4.0, as_bias=True, drop_rate=0.0, drop_path_rate=0.1, norm_layer=MyNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], shift_size=shift_size, mlp_ratio=self.mlp_ratio, as_bias=as_bias, drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):

    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, padding=patch_size // 2), nn.GELU(), nn.BatchNorm2d(dim))
        self.blocks = nn.Sequential(*[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'), nn.GELU(), nn.BatchNorm2d(dim))), nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim)) for i in range(depth)])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, n_classes))

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        out = self.classifier(embedding)
        return out


class ConvTokenizer(nn.Module):

    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(3, embedding_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(embedding_dim // 2), nn.ReLU(inplace=True), nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(embedding_dim // 2), nn.ReLU(inplace=True), nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(embedding_dim), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1)))

    def forward(self, x):
        return self.block(x)


class ConvStage(nn.Module):

    def __init__(self, num_blocks=2, embedding_dim_in=64, hidden_dim=128, embedding_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(nn.Conv2d(embedding_dim_in, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, embedding_dim_in, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(embedding_dim_in), nn.ReLU(inplace=True))
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(embedding_dim_in, embedding_dim_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class ConvMLPStage(nn.Module):

    def __init__(self, embedding_dim, dim_feedforward=2048, stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.connect = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=embedding_dim, bias=False)
        self.connect_norm = nn.LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.drop_path = DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0 else nn.Identity()

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect(self.connect_norm(src).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(nn.Module):

    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = nn.Conv2d(embedding_dim_in, embedding_dim_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)


class BasicStage(nn.Module):

    def __init__(self, num_blocks, embedding_dims, mlp_ratio=1, stochastic_depth_rate=0.1, downsample=True):
        super(BasicStage, self).__init__()
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_blocks)]
        for i in range(num_blocks):
            block = ConvMLPStage(embedding_dim=embedding_dims[0], dim_feedforward=int(embedding_dims[0] * mlp_ratio), stochastic_depth_rate=dpr[i])
            self.blocks.append(block)
        self.downsample_mlp = ConvDownsample(embedding_dims[0], embedding_dims[1]) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x


class ConvMLP(nn.Module):

    def __init__(self, depth, d_model, expansion_factor, channels=64, n_conv_blocks=3, classifier_head=True, num_classes=1000, *args, **kwargs):
        super(ConvMLP, self).__init__()
        assert len(depth) == len(expansion_factor) == len(expansion_factor), f'depth, d_model and expansion_factor must agree in size, {len(depth)}, {len(d_model)} and {len(expansion_factor)} passed.'
        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks, embedding_dim_in=channels, hidden_dim=d_model[0], embedding_dim_out=d_model[0])
        self.stages = nn.ModuleList()
        for i in range(0, len(depth)):
            stage = BasicStage(num_blocks=depth[i], embedding_dims=d_model[i:i + 2], mlp_ratio=expansion_factor[i], stochastic_depth_rate=0.1, downsample=i + 1 < len(depth))
            self.stages.append(stage)
        if classifier_head:
            self.norm = nn.LayerNorm(d_model[-1])
            self.head = nn.Linear(d_model[-1], num_classes)
        else:
            self.head = None
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        x = x.permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
        if self.head is None:
            return x
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class CycleFC(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=True):
        super(CycleFC, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = self.kernel_size[0] * self.kernel_size[1] // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - self.kernel_size[1] // 2
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - self.kernel_size[0] // 2
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) ->Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def extra_repr(self) ->str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class CycleMLP(nn.Module):

    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Downsample transition stage"""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)

    def forward(self, x: Tensor) ->Tensor:
        return self.norm(self.proj(x))


def basic_blocks(dim, index, layers, mlp_ratio=3.0, qkv_bias=False, qk_scale=None, attn_drop=0.0, drop_path_rate=0.0, skip_lam=1.0, mlp_fn=CycleMLP, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)
    return blocks


class CycleNet(nn.Module):
    """ CycleMLP Network """

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """ mmseg or mmdet `init_weight` """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    def __init__(self, elementary_axes_lengths: List[int], input_composite_axes: List[Tuple[List[int], List[int]]], reduced_elementary_axes: List[int], axes_permutation: List[int], added_axes: Dict[int, int], output_composite_axes: List[List[int]], ellipsis_position_in_lhs: Optional[int]=None):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.input_composite_axes: List[Tuple[List[int], List[int]]] = input_composite_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes
        self.axes_permutation: List[int] = axes_permutation
        self.added_axes: Dict[int, int] = added_axes
        self.reduced_elementary_axes: List[int] = reduced_elementary_axes
        self.ellipsis_position_in_lhs: int = ellipsis_position_in_lhs if ellipsis_position_in_lhs is not None else 10000


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other """

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise EinopsError('No need to create anonymous axis of length 1. Report this as an issue')
            else:
                raise EinopsError('Anonymous axis should have positive length, not {}'.format(self.value))

    def __repr__(self):
        return '{}-axis'.format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """

    def __init__(self, expression, *, allow_underscore: bool=False):
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[str] = set()
        self.has_non_unitary_anonymous_axes: bool = False
        self.composition = []
        if '.' in expression:
            if '...' not in expression:
                raise EinopsError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise EinopsError('Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True
        bracket_group = None

        def add_axis_name(x):
            if x is not None:
                if x in self.identifiers:
                    if not (allow_underscore and x == '_'):
                        raise EinopsError('Indexing expression contains duplicate dimension "{}"'.format(x))
                if x == _ellipsis:
                    self.identifiers.add(_ellipsis)
                    if bracket_group is None:
                        self.composition.append(_ellipsis)
                        self.has_ellipsis_parenthesized = False
                    else:
                        bracket_group.append(_ellipsis)
                        self.has_ellipsis_parenthesized = True
                else:
                    is_number = str.isdecimal(x)
                    if is_number and int(x) == 1:
                        if bracket_group is None:
                            self.composition.append([])
                        else:
                            pass
                        return
                    is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                    if not (is_number or is_axis_name):
                        raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
                    if is_number:
                        x = AnonymousAxis(x)
                    self.identifiers.add(x)
                    if is_number:
                        self.has_non_unitary_anonymous_axes = True
                    if bracket_group is None:
                        self.composition.append([x])
                    else:
                        bracket_group.append(x)
        current_identifier = None
        for char in expression:
            if char in '() ':
                add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise EinopsError('Axis composition is one-level (brackets inside brackets not allowed)')
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise EinopsError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise EinopsError("Unknown character '{}'".format(char))
        if bracket_group is not None:
            raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
        add_axis_name(current_identifier)

    def flat_axes_order(self) ->List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), 'does not work with ellipsis'
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) ->bool:
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool=False) ->Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, 'not a valid python identifier'
        elif name[0] == '_' or name[-1] == '_':
            if name == '_' and allow_underscore:
                return True, ''
            return False, 'axis name should should not start or end with underscore'
        else:
            if keyword.iskeyword(name):
                warnings.warn('It is discouraged to use axes names that are keywords: {}'.format(name), RuntimeWarning)
            if name in ['axis']:
                warnings.warn("It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning)
            return True, ''

    @staticmethod
    def check_axis_name(name: str) ->bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _reason = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


ReductionCallable = Callable[[Tensor, List[int]], Tensor]


Reduction = Union[str, ReductionCallable]


_reductions = 'min', 'max', 'sum', 'mean', 'prod'


_unknown_axis_length = -999999


@functools.lru_cache(256)
def _prepare_transformation_recipe(pattern: str, operation: Reduction, axes_lengths: Tuple[Tuple, ...]) ->TransformRecipe:
    """ Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left, rght = pattern.split('->')
    left = ParsedExpression(left)
    rght = ParsedExpression(rght)
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError('Ellipsis found in right side, but not left side of a pattern {}'.format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError('Ellipsis is parenthesis in the left side is not allowed: {}'.format(pattern))
    if operation == 'rearrange':
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError('Non-unitary anonymous axes are not supported in rearrange (exception is length 1)')
        if len(difference) > 0:
            raise EinopsError('Identifiers only on one side of expression (should be on both): {}'.format(difference))
    elif operation == 'repeat':
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the left side of repeat: {}'.format(difference))
        axes_without_size = set.difference({ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)}, {*left.identifiers, *(ax for ax, _ in axes_lengths)})
        if len(axes_without_size) > 0:
            raise EinopsError('Specify sizes for new axes in repeat: {}'.format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the right side of reduce {}: {}'.format(operation, difference))
    else:
        raise EinopsError('Unknown reduction {}. Expect one of {}.'.format(operation, _reductions))
    axis_name2known_length = OrderedDict()
    for composite_axis in left.composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)
    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}
    reduced_axes: List[int] = [position for axis, position in axis_name2position.items() if axis not in rght.identifiers]
    reduced_axes: List[int] = list(sorted(reduced_axes))
    for elementary_axis, axis_length in axes_lengths:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError('Invalid name for an axis', elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError('Axis {} is not used in transform'.format(elementary_axis))
        axis_name2known_length[elementary_axis] = axis_length
    input_axes_known_unknown = []
    for composite_axis in left.composition:
        known = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError('Could not infer sizes for {}'.format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown]))
    axis_position_after_reduction = {}
    for axis_name in itertools.chain(*left.composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)
    result_axes_grouping: List[List[int]] = []
    for composite_axis in rght.composition:
        if composite_axis == _ellipsis:
            result_axes_grouping.append(_ellipsis_not_in_parenthesis)
        else:
            result_axes_grouping.append([axis_name2position[axis] for axis in composite_axis])
    ordered_axis_right = list(itertools.chain(*rght.composition))
    axes_permutation = [axis_position_after_reduction[axis] for axis in ordered_axis_right if axis in left.identifiers]
    added_axes = {i: axis_name2position[axis_name] for i, axis_name in enumerate(ordered_axis_right) if axis_name not in left.identifiers}
    ellipsis_left = None if _ellipsis not in left.composition else left.composition.index(_ellipsis)
    return TransformRecipe(elementary_axes_lengths=list(axis_name2known_length.values()), input_composite_axes=input_axes_known_unknown, reduced_elementary_axes=reduced_axes, axes_permutation=axes_permutation, added_axes=added_axes, output_composite_axes=result_axes_grouping, ellipsis_position_in_lhs=ellipsis_left)


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()

    def __repr__(self):
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) ->TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, operation='rearrange', axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        return _apply_recipe(self._recipe, x, reduction_type='rearrange')


class TorchJitBackend:
    """
    Completely static backend that mimics part of normal backend functionality
    but restricted to torch stuff only
    """

    @staticmethod
    def reduce(x: torch.Tensor, operation: str, reduced_axes: List[int]):
        if operation == 'min':
            return x.amin(dim=reduced_axes)
        elif operation == 'max':
            return x.amax(dim=reduced_axes)
        elif operation == 'sum':
            return x.sum(dim=reduced_axes)
        elif operation == 'mean':
            return x.mean(dim=reduced_axes)
        elif operation == 'prod':
            for i in list(sorted(reduced_axes))[::-1]:
                x = x.prod(dim=i)
            return x
        else:
            raise NotImplementedError('Unknown reduction ', operation)

    @staticmethod
    def transpose(x, axes: List[int]):
        return x.permute(axes)

    @staticmethod
    def stack_on_zeroth_dimension(tensors: List[torch.Tensor]):
        return torch.stack(tensors)

    @staticmethod
    def tile(x, repeats: List[int]):
        return x.repeat(repeats)

    @staticmethod
    def add_axes(x, n_axes: int, pos2len: Dict[int, int]):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = torch.unsqueeze(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    @staticmethod
    def is_float_type(x):
        return x.dtype in [torch.float16, torch.float32, torch.float64]

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, shape: List[int]):
        return x.reshape(shape)


CookedRecipe = Tuple[List[int], List[int], List[int], Dict[int, int], List[int]]


def _product(sequence: List[int]) ->int:
    """ minimalistic product that works both with numbers and symbols. Supports empty lists """
    result = 1
    for element in sequence:
        result *= element
    return result


def is_ellipsis_not_in_parenthesis(group: List[int]) ->bool:
    if len(group) != 1:
        return False
    return group[0] == -999


def _reconstruct_from_shape_uncached(self: TransformRecipe, shape: List[int]) ->CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, keras, theano) and UnknownSize (keras, mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    if self.ellipsis_position_in_lhs != 10000:
        if len(shape) < len(self.input_composite_axes) - 1:
            raise EinopsError('Expected at least {} dimensions, got {}'.format(len(self.input_composite_axes) - 1, len(shape)))
    elif len(shape) != len(self.input_composite_axes):
        raise EinopsError('Expected {} dimensions, got {}'.format(len(self.input_composite_axes), len(shape)))
    ellipsis_shape: List[int] = []
    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
        before_ellipsis = input_axis
        after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
        if input_axis == self.ellipsis_position_in_lhs:
            assert len(known_axes) == 0 and len(unknown_axes) == 1
            unknown_axis, = unknown_axes
            ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
            for d in ellipsis_shape:
                if d is None:
                    raise EinopsError("Couldn't infer shape for one or more axes represented by ellipsis")
            total_dim_size: int = _product(ellipsis_shape)
            axes_lengths[unknown_axis] = total_dim_size
        else:
            if input_axis < self.ellipsis_position_in_lhs:
                length = shape[before_ellipsis]
            else:
                length = shape[after_ellipsis]
            known_product = 1
            for axis in known_axes:
                known_product *= axes_lengths[axis]
            if len(unknown_axes) == 0:
                if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                    raise EinopsError('Shape mismatch, {} != {}'.format(length, known_product))
            else:
                if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                    raise EinopsError("Shape mismatch, can't divide axis of length {} in chunks of {}".format(length, known_product))
                unknown_axis: int = unknown_axes[0]
                inferred_length: int = length // known_product
                axes_lengths[unknown_axis] = inferred_length
    init_shapes = axes_lengths[:len(axes_lengths) - len(self.added_axes)]
    final_shapes: List[int] = []
    for output_axis, grouping in enumerate(self.output_composite_axes):
        if is_ellipsis_not_in_parenthesis(grouping):
            final_shapes.extend(ellipsis_shape)
        else:
            lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
            final_shapes.append(_product(lengths))
    reduced_axes = self.reduced_elementary_axes
    axes_reordering = self.axes_permutation
    added_axes: Dict[int, int] = {pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()}
    return init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes


def apply_for_scriptable_torch(recipe: TransformRecipe, tensor: torch.Tensor, reduction_type: str) ->torch.Tensor:
    backend = TorchJitBackend
    init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor))
    tensor = backend.reshape(tensor, init_shapes)
    if len(reduced_axes) > 0:
        tensor = backend.reduce(tensor, operation=reduction_type, reduced_axes=reduced_axes)
    tensor = backend.transpose(tensor, axes_reordering)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=len(axes_reordering) + len(added_axes), pos2len=added_axes)
    return backend.reshape(tensor, final_shapes)


class Rearrange(RearrangeMixin, torch.nn.Module):

    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type='rearrange')

    def _apply_recipe(self, x):
        pass


class DynaMixerOp_w(nn.Module):

    def __init__(self, w, dim, hidden_dim, segment):
        super().__init__()
        self.segment = segment
        self.reshape = Rearrange('b h w (s d) -> b h s w d', s=segment)
        self.Wd = nn.ModuleList([nn.Linear(dim, hidden_dim) for i in range(segment)])
        self.attend = nn.Sequential(Rearrange('b h w (s d) -> b h s (w d)', s=segment), nn.Linear(int(hidden_dim * w), w * w), Rearrange('b h s (w1 w2) -> b h s w1 w2', w1=w), nn.Softmax(dim=-1))
        self.recover = Rearrange('b h s w d -> b h w (s d)', s=segment)
        self.proc = nn.Linear(dim, dim)

    def forward(self, x):
        input = x
        x_ = []
        for i in range(self.segment):
            x_.append(self.Wd[i](x))
        x_ = torch.cat(x_, -1)
        attn = self.attend(x_)
        input = self.reshape(input)
        x = torch.matmul(attn, input)
        x = self.recover(x)
        return self.proc(x)


class DynaMixerOp_h(nn.Module):

    def __init__(self, h, dim, hidden_dim, segment):
        super().__init__()
        self.segment = segment
        self.reshape = Rearrange('b h w (s d) -> b w s h d', s=segment)
        self.Wd = nn.ModuleList([nn.Linear(dim, hidden_dim) for i in range(segment)])
        self.attend = nn.Sequential(Rearrange('b h w (s d) -> b w s (h d)', s=segment), nn.Linear(int(hidden_dim * h), h * h), Rearrange('b w s (h1 h2) -> b w s h1 h2', h1=h), nn.Softmax(dim=-1))
        self.recover = Rearrange('b w s h d -> b h w (s d)', s=segment)
        self.proc = nn.Linear(dim, dim)

    def forward(self, x):
        input = x
        x_ = []
        for i in range(self.segment):
            x_.append(self.Wd[i](x))
        x_ = torch.cat(x_, -1)
        attn = self.attend(x_)
        input = self.reshape(input)
        x = torch.matmul(attn, input)
        x = self.recover(x)
        return self.proc(x)


class DynaBlock(nn.Module):

    def __init__(self, h, w, dim, hidden_dim_DMO=2, segment=8):
        super().__init__()
        self.proj_c = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.DynaMixerOp_w = DynaMixerOp_w(w, dim, hidden_dim_DMO, segment)
        self.DynaMixerOp_h = DynaMixerOp_h(h, dim, hidden_dim_DMO, segment)

    def forward(self, x):
        Y_c = self.proj_c(x)
        Y_h = self.DynaMixerOp_h(x)
        Y_w = self.DynaMixerOp_w(x)
        Y_out = Y_h + Y_w + Y_c
        Y_out = self.proj_o(Y_out)
        return Y_out


class DynaMLPBlock(nn.Module):

    def __init__(self, depth, h, w, dim, hidden_dim_DMO, segment, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.reshape = Rearrange('b c h w -> b h w c')
        self.recover = Rearrange('b h w c -> b c h w')
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, DynaBlock(h, w, dim, hidden_dim_DMO, segment)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=0.0))]))

    def forward(self, x):
        x = self.reshape(x)
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        x = self.recover(x)
        return x


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) ->TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, operation=self.reduction, axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        return _apply_recipe(self._recipe, x, reduction_type=self.reduction)


class Reduce(ReduceMixin, torch.nn.Module):

    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type=self.reduction)

    def _apply_recipe(self, x):
        pass


dynamlp_settings = {'T': [[7, 2], [192, 384], [4, 14], [8, 16], 3, 0.1, 2], 'M': [[7, 2], [256, 512], [7, 17], [8, 16], 3, 0.1, 2], 'L': [[7, 2], [256, 512], [9, 27], [8, 16], 3, 0.3, 8]}


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DynaMixer(nn.Module):

    def __init__(self, model_name: str='M', image_size=224, in_channels: int=3, num_classes: int=1000):
        super().__init__()
        assert model_name in dynamlp_settings.keys(), f'DynaMLP model name should be in {list(dynamlp_settings.keys())}'
        patch_size, embed_dims, depths, segment, mlp_ratio, dropout, hidden_dim_DMO = dynamlp_settings[model_name]
        image_height, image_width = pair(image_size)
        h = []
        w = []
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            try:
                h.append(int(h[-1] / ps[0]))
                w.append(int(w[-1] / ps[1]))
            except:
                h.append(int(image_height / ps[0]))
                w.append(int(image_width / ps[1]))
            assert image_height % (ps[0] * oldps[0]) == 0, 'image must be divisible by patch size'
            assert image_width % (ps[1] * oldps[1]) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        self.stage = len(patch_size)
        self.stages = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=patch_size[i], stride=patch_size[i]), DynaMLPBlock(depth=depths[i], h=h[i], w=w[i], dim=embed_dims[i], hidden_dim_DMO=hidden_dim_DMO, segment=segment[i], mlp_dim=embed_dims[i] * mlp_ratio, dropout=dropout)) for i in range(self.stage)])
        self.mlp_head = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(embed_dims[-1], num_classes))

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out


def _report_axes(axes: set, report_message: str):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


class _EinmixMixin:

    def __init__(self, pattern, weight_shape, bias_shape=None, **axes_lengths):
        """
        EinMix - Einstein summation with automated tensor management and axis packing/unpacking.

        EinMix is an advanced tool, helpful tutorial:
        https://github.com/arogozhnikov/einops/blob/master/docs/3-einmix-layer.ipynb

        Imagine taking einsum with two arguments, one of each input, and one - tensor with weights
        >>> einsum('time batch channel_in, channel_in channel_out -> time batch channel_out', input, weight)

        This layer manages weights for you, syntax highlights separate role of weight matrix
        >>> EinMix('time batch channel_in -> time batch channel_out', weight_shape='channel_in channel_out')
        But otherwise it is the same einsum under the hood.

        Simple linear layer with bias term (you have one like that in your framework)
        >>> EinMix('t b cin -> t b cout', weight_shape='cin cout', bias_shape='cout', cin=10, cout=20)
        There is restriction to mix the last axis. Let's mix along height
        >>> EinMix('h w c-> hout w c', weight_shape='h hout', bias_shape='hout', h=32, hout=32)
        Channel-wise multiplication (like one used in normalizations)
        >>> EinMix('t b c -> t b c', weight_shape='c', c=128)
        Separate dense layer within each head, no connection between different heads
        >>> EinMix('t b (head cin) -> t b (head cout)', weight_shape='head cin cout', ...)

        ... ah yes, you need to specify all dimensions of weight shape/bias shape in parameters.

        Use cases:
        - when channel dimension is not last, use EinMix, not transposition
        - patch/segment embeddings
        - when need only within-group connections to reduce number of weights and computations
        - perfect as a part of sequential models
        - next-gen MLPs (follow tutorial to learn more)

        Uniform He initialization is applied to weight tensor and encounters for number of elements mixed.

        Parameters
        :param pattern: transformation pattern, left side - dimensions of input, right side - dimensions of output
        :param weight_shape: axes of weight. Tensor od this shape is created, stored, and optimized in a layer
        :param bias_shape: axes of bias added to output.
        :param axes_lengths: dimensions of weight tensor
        """
        super().__init__()
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.axes_lengths = axes_lengths
        left_pattern, right_pattern = pattern.split('->')
        left = ParsedExpression(left_pattern)
        right = ParsedExpression(right_pattern)
        weight = ParsedExpression(weight_shape)
        _report_axes(set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}), 'Unrecognized identifiers on the right side of EinMix {}')
        if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
            raise EinopsError('Ellipsis is not supported in EinMix (right now)')
        if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
            raise EinopsError('Anonymous axes (numbers) are not allowed in EinMix')
        if '(' in weight_shape or ')' in weight_shape:
            raise EinopsError(f'Parenthesis is not allowed in weight shape: {weight_shape}')
        pre_reshape_pattern = None
        pre_reshape_lengths = None
        post_reshape_pattern = None
        if any(len(group) != 1 for group in left.composition):
            names = []
            for group in left.composition:
                names += group
            composition = ' '.join(names)
            pre_reshape_pattern = f'{left_pattern}->{composition}'
            pre_reshape_lengths = {name: length for name, length in self.axes_lengths.items() if name in names}
        if any(len(group) != 1 for group in right.composition):
            names = []
            for group in right.composition:
                names += group
            composition = ' '.join(names)
            post_reshape_pattern = f'{composition}->{right_pattern}'
        self._create_rearrange_layers(pre_reshape_pattern, pre_reshape_lengths, post_reshape_pattern, {})
        for axis in weight.identifiers:
            if axis not in axes_lengths:
                raise EinopsError('Dimension {} of weight should be specified'.format(axis))
        _report_axes(set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}), 'Axes {} are not used in pattern')
        _report_axes(set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}), 'Weight axes {} are redundant')
        if len(weight.identifiers) == 0:
            warnings.warn('EinMix: weight has no dimensions (means multiplication by a number)')
        _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
        _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
        if bias_shape is not None:
            if not isinstance(bias_shape, str):
                raise EinopsError('bias shape should be string specifying which axes bias depends on')
            bias = ParsedExpression(bias_shape)
            _report_axes(set.difference(bias.identifiers, right.identifiers), 'Bias axes {} not present in output')
            _report_axes(set.difference(bias.identifiers, set(axes_lengths)), 'Sizes not provided for bias axes {}')
            _bias_shape = []
            for axes in right.composition:
                for axis in axes:
                    if axis in bias.identifiers:
                        _bias_shape.append(axes_lengths[axis])
                    else:
                        _bias_shape.append(1)
        else:
            _bias_shape = None
            _bias_input_size = None
        weight_bound = (3 / _fan_in) ** 0.5
        bias_bound = (1 / _fan_in) ** 0.5
        self._create_parameters(_weight_shape, weight_bound, _bias_shape, bias_bound)
        mapping2letters = {*left.identifiers, *right.identifiers, *weight.identifiers}
        mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapping2letters)}

        def write_flat(axes: list):
            return ''.join(mapping2letters[axis] for axis in axes)
        self.einsum_pattern: str = '{},{}->{}'.format(write_flat(left.flat_axes_order()), write_flat(weight.flat_axes_order()), write_flat(right.flat_axes_order()))

    def _create_rearrange_layers(self, pre_reshape_pattern: Optional[str], pre_reshape_lengths: Optional[Dict], post_reshape_pattern: Optional[str], post_reshape_lengths: Optional[Dict]):
        raise NotImplementedError('Should be defined in framework implementations')

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        """ Shape and implementations """
        raise NotImplementedError('Should be defined in framework implementations')

    def __repr__(self):
        params = repr(self.pattern)
        params += f", '{self.weight_shape}'"
        if self.bias_shape is not None:
            params += f", '{self.bias_shape}'"
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)


class EinMix(_EinmixMixin, torch.nn.Module):

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = torch.nn.Parameter(torch.zeros(weight_shape).uniform_(-weight_bound, weight_bound), requires_grad=True)
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape).uniform_(-bias_bound, bias_bound), requires_grad=True)
        else:
            self.bias = None

    def _create_rearrange_layers(self, pre_reshape_pattern: Optional[str], pre_reshape_lengths: Optional[Dict], post_reshape_pattern: Optional[str], post_reshape_lengths: Optional[Dict]):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **pre_reshape_lengths)
        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **post_reshape_lengths)

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result


class SpatialGatingUnit(nn.Module):

    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):

    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):

    def __init__(self, d_model=256, d_ffn=1536, seq_len=256, depth=30):
        super().__init__()
        self.model = nn.Sequential(*[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(depth)])

    def forward(self, x):
        return self.model(x)


def check_sizes(image_size, patch_size):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image height and width must be divisible by patch size'
    num_patches = image_height // patch_height * (image_width // patch_width)
    return num_patches


class gMLPForImageClassification(gMLP):

    def __init__(self, image_size=256, patch_size=16, in_channels=3, num_classes=1000, d_model=256, d_ffn=1536, depth=30):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, num_patches, depth)
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size))
        self.mlp_head = nn.Sequential(nn.Linear(d_model, num_classes))

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PatchEmbedding(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, norm_layer=False):
        super().__init__()
        self.reduction = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding), nn.Identity() if not norm_layer else nn.Sequential(Rearrange('b c h w -> b h w c'), nn.LayerNorm(dim_out), Rearrange('b h w c -> b c h w')))

    def forward(self, x):
        return self.reduction(x)


class CrossRegion(nn.Module):

    def __init__(self, step=1, dim=1):
        super().__init__()
        self.step = step
        self.dim = dim

    def forward(self, x):
        return torch.roll(x, self.step, self.dim)


class InnerRegionW(nn.Module):

    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(Rearrange('b c h (w group) -> b (c w) h group', w=self.w))

    def forward(self, x):
        return self.region(x)


class InnerRegionH(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(Rearrange('b c (h group) w -> b (c h) group w', h=self.h))

    def forward(self, x):
        return self.region(x)


class InnerRegionRestoreW(nn.Module):

    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(Rearrange('b (c w) h group -> b c h (w group)', w=self.w))

    def forward(self, x):
        return self.region(x)


class InnerRegionRestoreH(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(Rearrange('b (c h) group w -> b c (h group) w', h=self.h))

    def forward(self, x):
        return self.region(x)


class HireMLPBlock(nn.Module):

    def __init__(self, h, w, d_model, cross_region_step=1, cross_region_id=0, cross_region_interval=2, padding_type='circular'):
        super().__init__()
        assert padding_type in ['constant', 'reflect', 'replicate', 'circular']
        self.padding_type = padding_type
        self.w = w
        self.h = h
        self.cross_region = cross_region_id % cross_region_interval == 0
        if self.cross_region:
            self.cross_regionW = CrossRegion(step=cross_region_step, dim=3)
            self.cross_regionH = CrossRegion(step=cross_region_step, dim=2)
            self.cross_region_restoreW = CrossRegion(step=-cross_region_step, dim=3)
            self.cross_region_restoreH = CrossRegion(step=-cross_region_step, dim=2)
        else:
            self.cross_regionW = nn.Identity()
            self.cross_regionH = nn.Identity()
            self.cross_region_restoreW = nn.Identity()
            self.cross_region_restoreH = nn.Identity()
        self.inner_regionW = InnerRegionW(w)
        self.inner_regionH = InnerRegionH(h)
        self.inner_region_restoreW = InnerRegionRestoreW(w)
        self.inner_region_restoreH = InnerRegionRestoreH(h)
        self.proj_h = FeedForward(h * d_model, d_model // 2, h * d_model)
        self.proj_w = FeedForward(w * d_model, d_model // 2, w * d_model)
        self.proj_c = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        padding_num_w = W % self.w
        padding_num_h = H % self.h
        x = nn.functional.pad(x, (0, self.w - padding_num_w, 0, self.h - padding_num_h), self.padding_type)
        x_h = self.inner_regionH(self.cross_regionH(x))
        x_w = self.inner_regionW(self.cross_regionW(x))
        x_h = self.proj_h(x_h)
        x_w = self.proj_w(x_w)
        x_c = self.proj_c(x)
        x_h = self.cross_region_restoreH(self.inner_region_restoreH(x_h))
        x_w = self.cross_region_restoreW(self.inner_region_restoreW(x_w))
        out = x_c + x_h + x_w
        out = out[:, :, 0:H, 0:W]
        out = out.permute(0, 2, 3, 1)
        return out


class HireMLPStage(nn.Module):

    def __init__(self, h, w, d_model_in, d_model_out, depth, cross_region_step, cross_region_interval, expansion_factor=2, dropout=0.0, pooling=False, padding_type='circular'):
        super().__init__()
        self.pooling = pooling
        self.patch_merge = nn.Sequential(Rearrange('b h w c -> b c h w'), PatchEmbedding(d_model_in, d_model_out, kernel_size=3, stride=2, padding=1, norm_layer=False), Rearrange('b c h w -> b h w c'))
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model_in, nn.Sequential(HireMLPBlock(h, w, d_model_in, cross_region_step=cross_region_step, cross_region_id=i_depth + 1, cross_region_interval=cross_region_interval, padding_type=padding_type)), norm=nn.LayerNorm), PreNormResidual(d_model_in, nn.Sequential(nn.Linear(d_model_in, d_model_in * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model_in * expansion_factor, d_model_in), nn.Dropout(dropout)), norm=nn.LayerNorm)) for i_depth in range(depth)])

    def forward(self, x):
        x = self.model(x)
        if self.pooling:
            x = self.patch_merge(x)
        return x


class HireMLP(nn.Module):

    def __init__(self, patch_size=4, in_channels=3, num_classes=1000, d_model=[64, 128, 320, 512], h=[4, 3, 3, 2], w=[4, 3, 3, 2], cross_region_step=[2, 2, 1, 1], cross_region_interval=2, depth=[4, 6, 24, 3], expansion_factor=2, patcher_norm=False, padding_type='circular'):
        patch_size = pair(patch_size)
        super().__init__()
        self.patcher = PatchEmbedding(dim_in=in_channels, dim_out=d_model[0], kernel_size=7, stride=patch_size, padding=3, norm_layer=patcher_norm)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            i_depth = depth[i_layer]
            i_stage = HireMLPStage(h[i_layer], w[i_layer], d_model[i_layer], d_model_out=d_model[i_layer + 1] if i_layer + 1 < len(depth) else d_model[-1], depth=i_depth, cross_region_step=cross_region_step[i_layer], cross_region_interval=cross_region_interval, expansion_factor=expansion_factor, pooling=i_layer + 1 < len(depth), padding_type=padding_type)
            self.layers.append(i_stage)
        self.mlp_head = nn.Sequential(nn.LayerNorm(d_model[-1]), Reduce('b h w c -> b c', 'mean'), nn.Linear(d_model[-1], num_classes))

    def forward(self, x):
        embedding = self.patcher(x)
        embedding = embedding.permute(0, 2, 3, 1)
        for layer in self.layers:
            embedding = layer(embedding)
        out = self.mlp_head(embedding)
        return out


class MLPMixer(nn.Module):

    def __init__(self, num_patches, d_model, depth, expansion_factor=4, dropout=0.0):
        super().__init__()
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, FeedForward(num_patches, num_patches * expansion_factor, dropout, chan_first)), PreNormResidual(d_model, FeedForward(d_model, d_model * expansion_factor, dropout, chan_last))) for _ in range(depth)])

    def forward(self, x):
        return self.model(x)


class MLPMixerForImageClassification(MLPMixer):

    def __init__(self, in_channels=3, d_model=512, num_classes=1000, patch_size=16, image_size=224, depth=12, expansion_factor=4):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(num_patches, d_model, depth, expansion_factor)
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size))
        self.active = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(nn.Linear(d_model, num_classes))

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = self.active(embedding)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out


class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim=None) ->None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) ->Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MorphFC(nn.Module):

    def __init__(self, L, C):
        super().__init__()
        assert C % L == 0
        self.L = L
        self.C = C
        self.D = int(self.C / self.L)
        self.reshape_h = Rearrange('b (D group_C) (L group_H) w -> b (D L) (group_C group_H) w', D=self.D, L=self.L)
        self.recover_h = Rearrange('b (D L) (group_C group_H) w -> b (D group_C) (L group_H) w', D=self.D, group_C=self.L)
        self.reshape_w = Rearrange('b (D group_C) h (L group_W) -> b (D L) h (group_C group_W)', D=self.D, L=self.L)
        self.recover_w = Rearrange('b (D L) h (group_C group_W) -> b (D group_C) h (L group_W)', D=self.D, group_C=self.L)
        self.fc_h = nn.Conv2d(C, C, 1)
        self.fc_w = nn.Conv2d(C, C, 1)
        self.fc_c = nn.Conv2d(C, C, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        need_padding_h = H % self.L > 0
        need_padding_w = W % self.L > 0
        P_l, P_r, P_t, P_b = (self.L - W % self.L) // 2, self.L - W % self.L - (self.L - W % self.L) // 2, (self.L - H % self.L) // 2, self.L - H % self.L - (self.L - H % self.L) // 2
        x_h = F.pad(x, [0, 0, P_t, P_b, 0, 0], 'constant', 0) if need_padding_h else x
        x_w = F.pad(x, [P_l, P_r, 0, 0, 0, 0], 'constant', 0) if need_padding_w else x
        x_h = self.fc_h(x_h)
        x_w = self.fc_w(x_w)
        x_c = self.fc_c(x)
        x_h = x_h[:, :, P_t:-P_b, :].contiguous() if need_padding_h else x_h
        x_w = x_w[:, :, :, P_l:-P_r].contiguous() if need_padding_w else x_w
        x = x_h + x_w + x_c
        return x


class PATM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc_h = nn.Conv2d(dim, dim, 1)
        self.fc_w = nn.Conv2d(dim, dim, 1)
        self.fc_c = nn.Conv2d(dim, dim, 1)
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), 1, (0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), 1, (7 // 2, 0), groups=dim, bias=False)
        self.reweight = MLP(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU())
        self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x: Tensor) ->Tensor:
        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        c = self.fc_c(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4, dpr=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = PATM(dim)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) ->Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """

    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) ->Tensor:
        return self.norm(self.proj(x))


morphmlp_settings = {'T': [[3, 4, 7, 3], [4, 4, 4, 4], [84, 168, 336, 588], [14, 28, 28, 49], [0.1, 0.1, 0.1, 0.1]], 'S': [[3, 4, 9, 3], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.1, 0.1, 0.1, 0.1]], 'B': [[4, 6, 15, 4], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.3, 0.3, 0.3, 0.3]], 'L': [[4, 8, 18, 6], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.4, 0.4, 0.4, 0.4]]}


class MorphMLP(nn.Module):

    def __init__(self, model_name: str='T', pretrained: str=None, num_classes: int=1000, *args, **kwargs) ->None:
        super().__init__()
        assert model_name in morphmlp_settings.keys(), f'WaveMLP model name should be in {list(morphmlp_settings.keys())}'
        layers, mlp_ratios, embed_dims, chunk_len, stoch_drop = morphmlp_settings[model_name]
        self.patch_embed = PatchEmbedOverlap(7, 4, 2, embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = nn.Sequential(*[Block(embed_dims[i], chunk_len[i], mlp_ratios[i], stoch_drop[i]) for _ in range(layers[i])])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i + 1]))
        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str=None) ->None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'])
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def return_features(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f'norm{i}')(x)
                outs.append(out)
        return outs

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for blk in self.network:
            x = blk(x)
        x = self.norm(x)
        x = self.head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MixShiftBlock(nn.Module):
    """ Mix-Shifting Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size, layer_scale_init_value=1e-06, mlp_ratio=4, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        self.shift_dist = shift_dist
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]
        self.kernel_size = [(ms, ms // 2) for ms in mix_size]
        self.dwconv_lr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        self.dwconv_td = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        B_, C, H, W = x.shape
        xs = torch.chunk(x, self.shift_size, 1)
        x_shift_lr = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
        x_shift_td = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]
        for i in range(self.shift_size):
            x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
            x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])
        x_lr = torch.cat(x_shift_lr, 1)
        x_td = torch.cat(x_shift_td, 1)
        x = x_lr + x_td
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        flops += N * self.dim
        flops += self.dim * H * W
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops


class MS_MLP(nn.Module):
    """ MS-MLP
        PyTorch impl of : `MS-MLP`  -
          https://arxiv.org/abs/2202.06510
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each MS-MLP layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.GroupNorm with group=1.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], shift_size=5, shift_dist=[-2, -1, 0, 1, 2], mix_size=[[1, 1, 3, 5, 7], [1, 1, 3, 5, 5], [1, 1, 3, 3, 3], [1, 1, 1, 1, 3]], mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, norm_layer=LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], shift_size=shift_size, shift_dist=shift_dist, mix_size=mix_size[i_layer], mlp_ratio=self.mlp_ratio, drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchEmbed if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class ChannelBlock(Block):

    def __init__(self, dim, expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.LayerNorm(dim)


class TokenBlock(Block):

    def __init__(self, dim, channels, expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(*[Rearrange('b c o -> b o c'), nn.LayerNorm(channels), Rearrange('b o c -> b c o')])


class SpatiallySeparatedTokenBlock(Block):

    def __init__(self, dim, channels, expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(*[Rearrange('b (c o1) o2 -> b (o1 o2) c', c=channels, o2=dim), nn.LayerNorm(channels), Rearrange('b (o1 o2) c -> b (c o1) o2', c=channels, o2=dim)])


class PermutedBlock(Block):

    def __init__(self, spatial_dim, channels, raft_size, expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(spatial_dim * raft_size, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(*[Rearrange('b (c1 o1) (c2 o2) -> b (o1 o2) (c1 c2)', c1=channels // raft_size, c2=raft_size, o2=spatial_dim), nn.LayerNorm(channels), Rearrange('b (o1 o2) (c1 c2) -> b (c1 o1) (c2 o2)', c1=channels // raft_size, c2=raft_size, o2=spatial_dim)])


class Level(nn.Module, ABC):

    def __init__(self, image_size=224, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fn = nn.Identity()
        self._bh = self._bw = image_size // patch_size
        self._h = self._w = math.ceil(image_size / patch_size)

    def forward(self, input):
        if not (self._bh == self._h and self._bw == self._w):
            input = F.interpolate(input, (self._h * self.patch_size, self._w * self.patch_size), mode='bilinear', align_corners=False)
        return self.fn(input)


class SeparatedLNCodimLevel(Level):

    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, token_expansion_factor=2, channel_expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_size ** 2 * in_channels, out_channels) if patch_size != 1 or patch_size == 1 and in_channels == out_channels else nn.Identity(), *[nn.Sequential(*[Rearrange('b (h w) c -> b (c w) h', h=self._h), TokenBlock(self._h, out_channels * self._w, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (c w) h -> b (c h) w', h=self._h, w=self._w), TokenBlock(self._w, out_channels * self._h, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (c h) w -> b (h w) c', h=self._h, w=self._w), ChannelBlock(out_channels, channel_expansion_factor, dropout, drop_path_rate)]) for _ in range(depth)], Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class SeparatedLNChannelLevel(Level):

    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, token_expansion_factor=2, channel_expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_size ** 2 * in_channels, out_channels) if patch_size != 1 or patch_size == 1 and in_channels == out_channels else nn.Identity(), *[nn.Sequential(*[Rearrange('b (h w) c -> b (c w) h', h=self._h), SpatiallySeparatedTokenBlock(self._h, out_channels, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (c w) h -> b (c h) w', h=self._h, w=self._w), SpatiallySeparatedTokenBlock(self._w, out_channels, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (c h) w -> b (h w) c', h=self._h, w=self._w), ChannelBlock(out_channels, channel_expansion_factor, dropout, drop_path_rate)]) for _ in range(depth)], Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class SerialPermutedLevel(Level):

    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, token_expansion_factor=2, channel_expansion_factor=4, dropout=0.0, drop_path_rate=0.0, raft_size=4):
        super().__init__(image_size, patch_size)
        assert out_channels % raft_size == 0
        self.fn = nn.Sequential(*[Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_size ** 2 * in_channels, out_channels) if patch_size != 1 or patch_size == 1 and in_channels == out_channels else nn.Identity(), *[nn.Sequential(*[Rearrange('b (h w) (chw co) -> b (co w) (chw h)', h=self._h, w=self._w, chw=raft_size), PermutedBlock(self._h, out_channels, raft_size, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (co w) (chw h) -> b (co h) (chw w)', h=self._h, w=self._w, chw=raft_size), PermutedBlock(self._w, out_channels, raft_size, token_expansion_factor, dropout, drop_path_rate), Rearrange('b (co h) (chw w) -> b (h w) (chw co)', h=self._h, w=self._w, chw=raft_size), ChannelBlock(out_channels, channel_expansion_factor, dropout, drop_path_rate)]) for _ in range(depth)], Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class OriginalLevel(Level):

    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, token_expansion_factor=2, channel_expansion_factor=4, dropout=0.0, drop_path_rate=0.0):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_size ** 2 * in_channels, out_channels), *[nn.Sequential(*[Rearrange('b (h w) c -> b c (h w)', h=self._h, w=self._w), TokenBlock(self._h * self._w, out_channels, token_expansion_factor, dropout, drop_path_rate), Rearrange('b c (h w) -> b (h w) c', h=self._h, w=self._w), ChannelBlock(out_channels, channel_expansion_factor, dropout, drop_path_rate)]) for _ in range(depth)], Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


DEPTH = 'depth'


DIM = 'dim'


ORIGINAL_TM = 'original_tm'


PATCH_SIZE = 'patch_size'


RAFT_SIZE = 'raft_size'


SEP_LN_CH_TM = 'sep_ln_ch_tm'


SEP_LN_CODIM_TM = 'sep_ln_codim_tm'


SER_PM = 'ser_pm'


TOKEN_MIXING_TYPES = [SER_PM, SEP_LN_CODIM_TM, SEP_LN_CH_TM, ORIGINAL_TM]


class RaftMLP(nn.Module):

    def __init__(self, layers: List[Dict], in_channels: int=3, image_size: int=224, num_classes: int=1000, token_expansion_factor: int=2, channel_expansion_factor: int=4, dropout: float=0.0, token_mixing_type: str=SER_PM, shortcut: bool=True, gap: bool=False, drop_path_rate: float=0.0):
        assert token_mixing_type in TOKEN_MIXING_TYPES
        for i, layer in enumerate(layers):
            assert DEPTH in layer
            assert DIM in layer
            assert PATCH_SIZE in layer
            assert token_mixing_type != SER_PM or RAFT_SIZE in layer
            assert 0 < layer.get(DIM)
        super().__init__()
        self.layers = layers
        self.shortcut = shortcut
        self.gap = gap
        if token_mixing_type == ORIGINAL_TM:
            level = OriginalLevel
        elif token_mixing_type == SEP_LN_CODIM_TM:
            level = SeparatedLNCodimLevel
        elif token_mixing_type == SEP_LN_CH_TM:
            level = SeparatedLNChannelLevel
        else:
            level = SerialPermutedLevel
        levels = []
        heads = []
        for i, layer in enumerate(self.layers):
            params = {'in_channels': in_channels if i == 0 else self.layers[i - 1].get(DIM), 'out_channels': layer.get(DIM), 'depth': layer.get(DEPTH), 'image_size': image_size, 'patch_size': layer.get(PATCH_SIZE), 'token_expansion_factor': token_expansion_factor, 'channel_expansion_factor': channel_expansion_factor, 'dropout': dropout, 'drop_path_rate': drop_path_rate}
            if token_mixing_type == SER_PM:
                params['raft_size'] = layer.get(RAFT_SIZE)
            levels.append(level(**params))
            heads_seq = []
            if self.shortcut or len(self.layers) == i + 1:
                heads_seq.append(Rearrange('b c h w -> b h w c'))
                heads_seq.append(nn.LayerNorm(layer.get(DIM)))
                heads_seq.append(Rearrange('b h w c -> b c h w'))
                if gap or len(self.layers) != i + 1:
                    heads_seq.append(Reduce('b c h w -> b c', 'mean'))
                if len(self.layers) != i + 1:
                    heads_seq.append(nn.Linear(layer.get(DIM), self.layers[-1].get(DIM) * 2))
                heads.append(nn.Sequential(*heads_seq))
            image_size = math.ceil(image_size / layer.get(PATCH_SIZE))
        self.levels = nn.ModuleList(levels)
        self.heads = nn.ModuleList(heads)
        self.classifier = nn.Linear(self.layers[-1].get(DIM) if gap else self.layers[-1].get(DIM) * image_size ** 2, num_classes)
        if not gap:
            self.flatten = nn.Flatten()

    def forward(self, input):
        output = []
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if self.shortcut:
                output.append(self.heads[i](input))
        if not self.shortcut:
            output = self.heads[0](input)
        else:
            output = reduce(lambda a, b: b[:, :self.layers[-1].get(DIM)] * a + b[:, self.layers[-1].get(DIM):], output[::-1]) if self.gap else reduce(lambda a, b: b[:, :self.layers[-1].get(DIM)].view(-1, self.layers[-1].get(DIM), 1, 1) * a + b[:, self.layers[-1].get(DIM):].view(-1, self.layers[-1].get(DIM), 1, 1), output[::-1])
        if not self.gap:
            output = self.flatten(output)
        return self.classifier(output)


class GlobalPerceptron(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)
    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(repeat_times, 0)


class RepMLPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, h, w, reparam_conv_k=None, globalperceptron_reduce=4, num_sharesets=1, deploy=False):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        self.S = num_sharesets
        self.h, self.w = h, w
        self.deploy = deploy
        assert in_channels == out_channels
        self.gp = GlobalPerceptron(input_channels=in_channels, internal_neurons=in_channels // globalperceptron_reduce)
        self.fc3 = nn.Conv2d(self.h * self.w * num_sharesets, self.h * self.w * num_sharesets, 1, 1, 0, bias=deploy, groups=num_sharesets)
        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets)
        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = conv_bn(num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k // 2, groups=num_sharesets)
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.S, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
        return out

    def forward(self, inputs):
        global_vec = self.gp(inputs)
        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w
        partitions = self.partition(inputs, h_parts, w_parts)
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.S, self.h, self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
            fc3_out += conv_out
        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.S * self.h * self.w, self.S * self.h * self.w, 1, 1, 0, bias=True, groups=self.S)
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = torch.eye(self.h * self.w).repeat(1, self.S).reshape(self.h * self.w, self.S, self.h, self.w)
        fc_k = F.conv2d(I, conv_kernel, padding=(conv_kernel.size(2) // 2, conv_kernel.size(3) // 2), groups=self.S)
        fc_k = fc_k.reshape(self.h * self.w, self.S * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


class FFNBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
        self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


class RepMLPNetUnit(nn.Module):

    def __init__(self, channels, h, w, reparam_conv_k, globalperceptron_reduce, ffn_expand=4, num_sharesets=1, deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(in_channels=channels, out_channels=channels, h=h, w=w, reparam_conv_k=reparam_conv_k, globalperceptron_reduce=globalperceptron_reduce, num_sharesets=num_sharesets, deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
    result.add_module('relu', nn.ReLU())
    return result


class RepMLPNet(nn.Module):

    def __init__(self, in_channels=3, num_class=1000, patch_size=(4, 4), num_blocks=(2, 2, 6, 2), channels=(192, 384, 768, 1536), hs=(64, 32, 16, 8), ws=(64, 32, 16, 8), sharesets_nums=(4, 8, 16, 32), reparam_conv_k=(3,), globalperceptron_reduce=4, use_checkpoint=False, deploy=False):
        super().__init__()
        num_stages = len(num_blocks)
        assert num_stages == len(channels)
        assert num_stages == len(hs)
        assert num_stages == len(ws)
        assert num_stages == len(sharesets_nums)
        self.conv_embedding = conv_bn_relu(in_channels, channels[0], kernel_size=patch_size, stride=patch_size, padding=0)
        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [RepMLPNetUnit(channels=channels[stage_idx], h=hs[stage_idx], w=ws[stage_idx], reparam_conv_k=reparam_conv_k, globalperceptron_reduce=globalperceptron_reduce, ffn_expand=4, num_sharesets=sharesets_nums[stage_idx], deploy=deploy) for _ in range(num_blocks[stage_idx])]
            stages.append(nn.ModuleList(stage_blocks))
            if stage_idx < num_stages - 1:
                embeds.append(conv_bn_relu(in_channels=channels[stage_idx], out_channels=channels[stage_idx + 1], kernel_size=2, stride=2, padding=0))
        self.stages = nn.ModuleList(stages)
        self.embeds = nn.ModuleList(embeds)
        self.head_norm = nn.BatchNorm2d(channels[-1])
        self.head = nn.Linear(channels[-1], num_class)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        x = self.conv_embedding(x)
        for i, stage in enumerate(self.stages):
            for block in stage:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(embed, x)
                else:
                    x = embed(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def locality_injection(self):
        for m in self.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()


class Aff(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class MLPblock(nn.Module):

    def __init__(self, num_patch, dim, mlp_dim, dropout=0.0, depth=18):
        super().__init__()
        if depth <= 18:
            init_values = 0.1
        elif depth > 18 and depth <= 24:
            init_values = 1e-05
        else:
            init_values = 1e-06
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Conv1d(num_patch, num_patch, kernel_size=1)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x


class ResMLP(nn.Module):

    def __init__(self, num_patch, d_model, depth, expansion_factor):
        super().__init__()
        self.model = nn.Sequential(*[MLPblock(num_patch, d_model, d_model * expansion_factor, depth=depth) for _ in range(depth)])

    def forward(self, x):
        return self.model(x)


class ResMLPForImageClassification(ResMLP):

    def __init__(self, in_channels=3, d_model=384, num_classes=1000, patch_size=16, image_size=224, depth=12, expansion_factor=4):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(num_patches, d_model, depth, expansion_factor)
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size))
        self.affine = Aff(d_model)
        self.mlp_head = nn.Sequential(nn.Linear(d_model, num_classes))

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out


class Spatial_Shift(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, w, h, c = x.size()
        x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
        x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
        x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
        x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
        return x


class SplitAttention(nn.Module):

    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, h, w, c = x.size()
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        return x


class S2Block(nn.Module):

    def __init__(self, d_model, depth, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, S2Attention(d_model)), PreNormResidual(d_model, nn.Sequential(nn.Linear(d_model, d_model * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * expansion_factor, d_model), nn.Dropout(dropout)))) for _ in range(depth)])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x


class S2MLPv1(nn.Module):

    def __init__(self, image_size=224, patch_size=[7, 2], in_channels=3, num_classes=1000, d_model=[192, 384], depth=[4, 14], expansion_factor=[3, 3]):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert image_size[0] % (ps[0] * oldps[0]) == 0, 'image must be divisible by patch size'
            assert image_size[1] % (ps[1] * oldps[1]) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert len(patch_size) == len(depth) == len(d_model) == len(expansion_factor), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()
        self.stage = len(patch_size)
        self.stages = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=patch_size[i], stride=patch_size[i]), S2Block(d_model[i], depth[i], expansion_factor[i], dropout=0.0)) for i in range(self.stage)])
        self.mlp_head = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(d_model[-1], num_classes))

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out


class S2MLPv2(nn.Module):

    def __init__(self, image_size=224, patch_size=[7, 2], in_channels=3, num_classes=1000, d_model=[192, 384], depth=[4, 14], expansion_factor=[3, 3]):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert image_size[0] % (ps[0] * oldps[0]) == 0, 'image must be divisible by patch size'
            assert image_size[1] % (ps[1] * oldps[1]) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert len(patch_size) == len(depth) == len(d_model) == len(expansion_factor), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()
        self.stage = len(patch_size)
        self.stages = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=patch_size[i], stride=patch_size[i]), S2Block(d_model[i], depth[i], expansion_factor[i], dropout=0.0)) for i in range(self.stage)])
        self.mlp_head = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(d_model[-1], num_classes))

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out


class BiLSTM2D(nn.Module):

    def __init__(self, d_model, hidden_d_model):
        super().__init__()
        self.rnn_v = nn.LSTM(d_model, hidden_d_model, num_layers=1, batch_first=True, bias=True, bidirectional=True)
        self.rnn_h = nn.LSTM(d_model, hidden_d_model, num_layers=1, batch_first=True, bias=True, bidirectional=True)
        self.fc = nn.Linear(4 * hidden_d_model, d_model)

    def forward(self, x):
        B, H, W, C = x.shape
        v, _ = self.rnn_v(x.permute(0, 2, 1, 3).reshape(-1, H, C))
        v = v.reshape(B, W, H, -1).permute(0, 2, 1, 3)
        h, _ = self.rnn_h(x.reshape(-1, W, C))
        h = h.reshape(B, H, W, -1)
        x = torch.cat([v, h], dim=-1)
        x = self.fc(x)
        return x


class Sequencer2DBlock(nn.Module):

    def __init__(self, d_model, depth, hidden_d_model, expansion_factor=3, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, nn.Sequential(BiLSTM2D(d_model, hidden_d_model))), PreNormResidual(d_model, nn.Sequential(nn.Linear(d_model, d_model * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * expansion_factor, d_model), nn.Dropout(dropout)))) for _ in range(depth)])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x


sequencer_settings = {'S': [[4, 3, 8, 3], [192, 384, 384, 384], [48, 96, 96, 96], 3], 'M': [[4, 3, 14, 3], [192, 384, 384, 384], [48, 96, 96, 96], 3], 'L': [[8, 8, 16, 4], [192, 384, 384, 384], [48, 96, 96, 96], 3]}


class Sequencer2D(nn.Module):

    def __init__(self, model_name: str='M', pretrained: str=None, num_classes: int=1000, in_channels=3, *args, **kwargs) ->None:
        super().__init__()
        assert model_name in sequencer_settings.keys(), f'Sequencer model name should be in {list(sequencer_settings.keys())}'
        depth, embed_dims, hidden_dims, expansion_factor = sequencer_settings[model_name]
        self.patch_size = [7, 2, 1, 1]
        self.stage = len(depth)
        self.stages = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=self.patch_size[i], stride=self.patch_size[i]), Sequencer2DBlock(embed_dims[i], depth[i], hidden_dims[i], expansion_factor, dropout=0.0)) for i in range(self.stage)])
        self.mlp_head = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(embed_dims[-1], num_classes))

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out


class sMLPBlock(nn.Module):

    def __init__(self, h=224, w=224, d_model=3):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Conv2d(3 * d_model, d_model, kernel_size=1)

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse)
        return out


class sMLPStage(nn.Module):

    def __init__(self, height, width, d_model, depth, expansion_factor=2, dropout=0.0, pooling=False):
        super().__init__()
        self.pooling = pooling
        self.patch_merge = nn.Sequential(Rearrange('b c h w -> b h w c'), PatchMerging((height, width), d_model), Rearrange('b h w c -> b c h w'))
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)), norm=nn.BatchNorm2d), PreNormResidual(d_model, nn.Sequential(sMLPBlock(height, width, d_model)), norm=nn.BatchNorm2d), Rearrange('b c h w -> b h w c'), PreNormResidual(d_model, nn.Sequential(nn.Linear(d_model, d_model * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * expansion_factor, d_model), nn.Dropout(dropout)), norm=nn.LayerNorm), Rearrange('b h w c -> b c h w')) for _ in range(depth)])

    def forward(self, x):
        x = self.model(x)
        if self.pooling:
            x = self.patch_merge(x)
        return x


class SparseMLP(nn.Module):

    def __init__(self, image_size=224, patch_size=4, in_channels=3, num_classes=1000, d_model=96, depth=[2, 10, 24, 2], expansion_factor=2, patcher_norm=False):
        image_size = pair(image_size)
        patch_size = pair(patch_size)
        assert image_size[0] % patch_size[0] == 0, 'image must be divisible by patch size'
        assert image_size[1] % patch_size[1] == 0, 'image must be divisible by patch size'
        height = image_size[0] // patch_size[0]
        width = image_size[1] // patch_size[1]
        super().__init__()
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size), nn.Identity() if not patcher_norm else nn.Sequential(Rearrange('b c h w -> b h w c'), nn.LayerNorm(d_model), Rearrange('b h w c -> b c h w')))
        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            i_depth = depth[i_layer]
            i_stage = sMLPStage(height // 2 ** i_layer, width // 2 ** i_layer, d_model, i_depth, expansion_factor=expansion_factor, pooling=i_layer + 1 < len(depth))
            self.layers.append(i_stage)
            if i_layer + 1 < len(depth):
                d_model = d_model * 2
        self.mlp_head = nn.Sequential(Rearrange('b c h w -> b h w c'), nn.LayerNorm(d_model), Reduce('b h w c -> b c', 'mean'), nn.Linear(d_model, num_classes))

    def forward(self, x):
        i = 0
        embedding = self.patcher(x)
        for layer in self.layers:
            i += 1
            embedding = layer(embedding)
        out = self.mlp_head(embedding)
        return out


class SwinMLP(nn.Module):
    """ Swin MLP
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


def dcn_v2_conv_backward(input, offset, mask, weight, bias, grad_output, stride, padding, dilation, deformable_groups):
    kernel_size = weight.shape[2:4]
    batch = input.shape[0]
    channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    channels_out = weight.shape[0]
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    ones = jt.ones((batch, height_out, width_out), dtype=input.dtype)
    colums = jt.empty((batch, channels * kernel_h * kernel_w, 1 * height_out * width_out), dtype=input.dtype)
    inputs = [input, weight, bias, offset, mask, ones, colums, grad_output]
    output_shape = [input.shape, weight.shape, bias.shape, offset.shape, mask.shape]
    output_type = [input.dtype, weight.dtype, bias.dtype, offset.dtype, mask.dtype]
    input_grad, weight_grad, bias_grad, offset_grad, mask_grad = jt.code(output_shape, output_type, inputs, cuda_header="""
#include<cstdio>
#include<cstring>
#include<algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
namespace jittor {
extern cublasHandle_t cublas_handle;
} // jittor
#define CUDA_KERNEL_LOOP(i, n)                          \\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \\
      i < (n);                                          \\
      i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}
__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
                                            const int data_width, const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  float weight = 0;
  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  return weight;
}
__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    const int c_col = c_im * kernel_h * kernel_w;
    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset, const float *data_mask,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output
    const int deformable_group_index = c / channel_per_deformable_group;
    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const float cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}
__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *data_col, const float *data_im,
                                                             const float *data_offset, const float *data_mask,
                                                             const int channels, const int height, const int width,
                                                             const int kernel_h, const int kernel_w,
                                                             const int pad_h, const int pad_w,
                                                             const int stride_h, const int stride_w,
                                                             const int dilation_h, const int dilation_w,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels, const int deformable_group,
                                                             const int height_col, const int width_col,
                                                             float *grad_offset, float *grad_mask)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output
    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;
    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;
      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      else
      {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const float weight = dmcn_get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}
void modulated_deformable_im2col_cuda(
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
}
void modulated_deformable_col2im_cuda(
  const float* data_col, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group, float* grad_im){
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, height_col, width_col, grad_im);
}
void modulated_deformable_col2im_coord_cuda(
  const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group,
  float* grad_offset, float* grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col, 
        grad_offset, grad_mask);
}
    """, cuda_src=f"""
    const int kernel_h = {kernel_h};
    const int kernel_w = {kernel_w};
    const int stride_h = {stride_h};
    const int stride_w = {stride_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int dilation_h = {dilation_h};
    const int dilation_w = {dilation_w};
    const int deformable_group = {deformable_groups};
    """ + """
    @alias(input,in0)
    @alias(weight,in1)
    @alias(bias,in2)
    @alias(offset,in3)
    @alias(mask,in4)
    @alias(ones,in5)
    @alias(columns,in6)
    @alias(grad_output,in7)
    @alias(grad_input,out0)
    @alias(grad_weight,out1)
    @alias(grad_bias,out2)
    @alias(grad_offset,out3)
    @alias(grad_mask,out4)
    const int batch = input_shape0;
    const int channels = input_shape1;
    const int height = input_shape2;
    const int width = input_shape3;
    const int channels_out = weight_shape0;
    const int channels_kernel = weight_shape1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    for (int b = 0; b < batch; b++)
    {
        auto input_n = input_p+input_stride0*b;
        auto offset_n = offset_p+offset_stride0*b;
        auto mask_n = mask_p+mask_stride0*b;
        auto grad_output_n = grad_output_p+grad_output_stride0*b;
        auto grad_input_n = grad_input_p+grad_input_stride0*b;
        auto grad_offset_n = grad_offset_p+grad_offset_stride0*b;
        auto grad_mask_n = grad_mask_p+grad_mask_stride0*b;
        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;
        float alpha0  = 1.0f;
        float beta0 = 0.0f;
        cublasHandle_t& handle = cublas_handle;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha0,
                         grad_output_n, n,
                         weight_p, m, &beta0,
                         columns_p, n);
        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(columns_p,
                                               input_n,
                                               offset_n,
                                               mask_n,
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n,
                                               grad_mask_n);
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(columns_p,
                                         offset_n,
                                         mask_n,
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n);
        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(
                                         input_n,
                                         offset_n,
                                         mask_n,
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns_p);
        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;
        float alpha  = 1.0f;
        float beta = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_, m_, k_, &alpha,
                         columns_p, k_,
                         grad_output_n, k_, &beta,
                         grad_weight_p, n_);
        //cublasDestroy(handle);
        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
        cublasSgemv(handle,
                         CUBLAS_OP_T,
                         k_, m_, &alpha,
                         grad_output_n, k_,
                         ones_p, 1, &beta,
                         grad_bias_p, 1);
    }
    """)
    return input_grad, offset_grad, mask_grad, weight_grad, bias_grad


def dcn_v2_conv_forward(input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups):
    kernel_size = weight.shape[2:4]
    batch = input.shape[0]
    channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    channels_out = weight.shape[0]
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    ones = jt.ones((batch, height_out, width_out), dtype=input.dtype)
    colums = jt.empty((batch, channels * kernel_h * kernel_w, 1 * height_out * width_out), dtype=input.dtype)
    inputs = [input, weight, bias, offset, mask, ones, colums]
    output_shape = batch, channels_out, height_out, width_out
    output_type = input.dtype
    output = jt.code(output_shape, output_type, inputs, cuda_header="\n#undef out\n#include<cstdio>\n#include<cstring>\n#include<algorithm>\n#include <cuda_runtime.h>\n#include <cublas_v2.h>\n#include <executor.h>\nusing namespace std;\nnamespace jittor {\nextern cublasHandle_t cublas_handle;\n} // jittor\n#define CUDA_KERNEL_LOOP(i, n)                          \\\n  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \\\n      i < (n);                                          \\\n      i += blockDim.x * gridDim.x)\nconst int CUDA_NUM_THREADS = 1024;\ninline int GET_BLOCKS(const int N)\n{\n  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;\n}\n__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,\n                                      const int height, const int width, float h, float w)\n{\n  int h_low = floor(h);\n  int w_low = floor(w);\n  int h_high = h_low + 1;\n  int w_high = w_low + 1;\n  float lh = h - h_low;\n  float lw = w - w_low;\n  float hh = 1 - lh, hw = 1 - lw;\n  float v1 = 0;\n  if (h_low >= 0 && w_low >= 0)\n    v1 = bottom_data[h_low * data_width + w_low];\n  float v2 = 0;\n  if (h_low >= 0 && w_high <= width - 1)\n    v2 = bottom_data[h_low * data_width + w_high];\n  float v3 = 0;\n  if (h_high <= height - 1 && w_low >= 0)\n    v3 = bottom_data[h_high * data_width + w_low];\n  float v4 = 0;\n  if (h_high <= height - 1 && w_high <= width - 1)\n    v4 = bottom_data[h_high * data_width + w_high];\n  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;\n  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);\n  return val;\n}\n__global__ void modulated_deformable_im2col_gpu_kernel(const int n,\n                                                       const float *data_im, const float *data_offset, const float *data_mask,\n                                                       const int height, const int width, const int kernel_h, const int kernel_w,\n                                                       const int pad_h, const int pad_w,\n                                                       const int stride_h, const int stride_w,\n                                                       const int dilation_h, const int dilation_w,\n                                                       const int channel_per_deformable_group,\n                                                       const int batch_size, const int num_channels, const int deformable_group,\n                                                       const int height_col, const int width_col,\n                                                       float *data_col)\n{\n  // launch channels * batch_size * height_col * width_col cores\n  CUDA_KERNEL_LOOP(index, n)\n  {\n    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)\n    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis\n    // index index of output matrix\n    const int w_col = index % width_col;\n    const int h_col = (index / width_col) % height_col;\n    // const int b_col = (index / width_col / height_col) % batch_size;\n    const int b_col = (index / width_col / height_col / num_channels) % batch_size;\n    // const int c_im = (index / width_col / height_col) / batch_size;\n    const int c_im = (index / width_col / height_col) % num_channels;\n    // const int c_col = c_im * kernel_h * kernel_w;\n    const int c_col = c_im * kernel_h * kernel_w;\n    // compute deformable group index\n    const int deformable_group_index = c_im / channel_per_deformable_group;\n    const int h_in = h_col * stride_h - pad_h;\n    const int w_in = w_col * stride_w - pad_w;\n    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;\n    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;\n    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;\n    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;\n    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;\n    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;\n    for (int i = 0; i < kernel_h; ++i)\n    {\n      for (int j = 0; j < kernel_w; ++j)\n      {\n        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;\n        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;\n        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;\n        const float offset_h = data_offset_ptr[data_offset_h_ptr];\n        const float offset_w = data_offset_ptr[data_offset_w_ptr];\n        const float mask = data_mask_ptr[data_mask_hw_ptr];\n        float val = static_cast<float>(0);\n        const float h_im = h_in + i * dilation_h + offset_h;\n        const float w_im = w_in + j * dilation_w + offset_w;\n        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {\n        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)\n        {\n          //const float map_h = i * dilation_h + offset_h;\n          //const float map_w = j * dilation_w + offset_w;\n          //const int cur_height = height - h_in;\n          //const int cur_width = width - w_in;\n          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);\n          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);\n        }\n        *data_col_ptr = val * mask;\n        // data_col_ptr += batch_size * height_col * width_col;\n        data_col_ptr += height_col * width_col;\n      }\n    }\n  }\n}\n__global__ void createBatchGemmBuffer(const float **input_b, float **output_b,\n                                      float **columns_b, const float **ones_b,\n                                      const float **weight_b, const float **bias_b,\n                                      float *input, float *output,\n                                      float *columns, float *ones,\n                                      float *weight, float *bias,\n                                      const int input_stride, const int output_stride,\n                                      const int columns_stride, const int ones_stride,\n                                      const int num_batches)\n{\n    const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < num_batches)\n    {\n        input_b[idx] = input + idx * input_stride;\n        output_b[idx] = output + idx * output_stride;\n        columns_b[idx] = columns + idx * columns_stride;\n        ones_b[idx] = ones + idx * ones_stride;\n        // share weights and bias within a Mini-Batch\n        weight_b[idx] = weight;\n        bias_b[idx] = bias;\n    }\n}\nvoid modulated_deformable_im2col_cuda(\n  const float* data_im, const float* data_offset, const float* data_mask,\n  const int batch_size, const int channels, const int height_im, const int width_im, \n  const int height_col, const int width_col, const int kernel_h, const int kernel_w,\n  const int pad_h, const int pad_w, const int stride_h, const int stride_w, \n  const int dilation_h, const int dilation_w,\n  const int deformable_group, float* data_col) {\n  // num_axes should be smaller than block size\n  const int channel_per_deformable_group = channels / deformable_group;\n  const int num_kernels = channels * batch_size * height_col * width_col;\n  modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(\n      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,\n      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,\n      batch_size, channels, deformable_group, height_col, width_col, data_col);\n}\n    ", cuda_src=f"""
    const int kernel_h = {kernel_h};
    const int kernel_w = {kernel_w};
    const int stride_h = {stride_h};
    const int stride_w = {stride_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int dilation_h = {dilation_h};
    const int dilation_w = {dilation_w};
    const int deformable_group = {deformable_groups};
""" + """
     @alias(input,in0)
    @alias(weight,in1)
    @alias(bias,in2)
    @alias(offset,in3)
    @alias(mask,in4)
    @alias(ones,in5)
    @alias(columns,in6)
    @alias(output,out0)
    const int batch = input_shape0;
    const int channels = input_shape1;
    const int height = input_shape2;
    const int width = input_shape3;
    const int channels_out = weight_shape0;
    const int channels_kernel = weight_shape1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    // prepare for batch-wise computing, which is significantly faster than instance-wise computing
    // when batch size is large.
    // launch batch threads
    int matrices_size = batch * sizeof(float *);
    const float ** input_b;
    float ** output_b;
    float ** columns_b;
    const float ** ones_b;
    const float ** weight_b;
    const float ** bias_b;
    size_t input_b_allocation;
    size_t output_b_allocation;
    size_t columns_b_allocation;
    size_t ones_b_allocation;
    size_t weight_b_allocation;
    size_t bias_b_allocation;
    input_b = (const float **)exe.allocator->alloc(matrices_size, input_b_allocation);
    output_b = (float **)exe.allocator->alloc(matrices_size, output_b_allocation);
    columns_b = (float **)exe.allocator->alloc(matrices_size, columns_b_allocation);
    ones_b = (const float **)exe.allocator->alloc(matrices_size, ones_b_allocation);
    weight_b = (const float **)exe.allocator->alloc(matrices_size, weight_b_allocation);
    bias_b = (const float **)exe.allocator->alloc(matrices_size, bias_b_allocation);
    const int block = 128;
    const int grid = (batch + block - 1) / block;
    createBatchGemmBuffer<<<grid, block>>>(
        input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        input_p,
        output_p,
        columns_p,
        ones_p,
        weight_p,
        bias_p,
        channels * width * height,
        channels_out * width_out * height_out,
        channels * kernel_h * kernel_w * height_out * width_out,
        height_out * width_out,
        batch);
    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    cublasHandle_t& handle = cublas_handle;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemmBatched(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            n_,
                            m_,
                            k_,
                            &alpha,
                            ones_b, k_,
                            bias_b, k_,
                            &beta,
                            output_b, n_,
                            batch);
    modulated_deformable_im2col_cuda(input_p,
                                     offset_p,
                                     mask_p,
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                     deformable_group,
                                     columns_p);
    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;
    float beta2 = 1.0f;
    cublasSgemmBatched(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            (const float **)columns_b, n,
                            weight_b, k,
                            &beta2,
                            output_b, n,
                            batch);
    exe.allocator->free(input_b, matrices_size, input_b_allocation);
    exe.allocator->free(output_b, matrices_size, output_b_allocation);
    exe.allocator->free(columns_b, matrices_size, columns_b_allocation);
    exe.allocator->free(ones_b, matrices_size, ones_b_allocation);
    exe.allocator->free(weight_b, matrices_size, weight_b_allocation);
    exe.allocator->free(bias_b, matrices_size, bias_b_allocation);
""")
    return output


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = jt.zeros((out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = jt.zeros((out_channels,))
        else:
            self.bias = np.zeros((out_channels,))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        nn.init.uniform_(self.weight, -stdv, stdv)

    def execute(self, x, offset):
        assert x.size(2) > self.kernel_size[0] and x.size(3) > self.kernel_size[1]
        mask_shape = list(offset.size())
        mask_shape[1] //= 2
        mask = jt.ones(mask_shape, x.dtype)
        return dcn_v2_conv(x, offset, mask, self.weight, jt.array(self.bias), self.stride, self.padding, self.dilation, self.deformable_groups)


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = jt.zeros((out_channels, in_channels, *self.kernel_size))
        self.bias = jt.zeros((out_channels,))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        nn.init.constant_(self.bias, 0.0)
        nn.init.uniform_(self.weight, -stdv, stdv)

    def execute(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


class Registry:

    def __init__(self):
        self._modules = {}

    def register_module(self, name=None, module=None):

        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules, f'{key} is already registered.'
            self._modules[key] = module
            return module
        if module is not None:
            return _register_module(module)
        return _register_module

    def get(self, name):
        assert name in self._modules, f'{name} is not registered.'
        return self._modules[name]


HEADS = Registry()


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0.0)
        nn.init.constant_(self.conv_offset_mask.bias, 0.0)

    def execute(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = jt.chunk(out, 3, dim=1)
        offset = jt.contrib.concat((o1, o2), dim=1)
        mask = jt.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


def dcn_v2_pooling_backward(grad_output, input, bbox, trans, output_count, no_trans, spatial_scale, output_dim, group_size, pooled_size, part_size, sample_per_part, trans_std):
    output_shape = [input.shape, trans.shape]
    output_dtype = [grad_output.dtype, trans.dtype]
    inputs = [grad_output, input, bbox, trans, output_count]
    input_grad, trans_grad = jt.code(output_shape, output_dtype, inputs, cuda_header='\n#include<cstdio>\n#include<cstring>\n#include<algorithm>\nusing namespace std;\n#define CUDA_KERNEL_LOOP(i, n)                        \\\n  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \\\n       i < (n);                                       \\\n       i += blockDim.x * gridDim.x)\nconst int CUDA_NUM_THREADS = 1024;\ninline int GET_BLOCKS(const int N)\n{\n  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;\n}\n__global__ void DeformablePSROIPoolBackwardAccKernel(\n    const int count,\n    const float *top_diff,\n    const float *top_count,\n    const int num_rois,\n    const float spatial_scale,\n    const int channels,\n    const int height, const int width,\n    const int pooled_height, const int pooled_width,\n    const int output_dim,\n    float *bottom_data_diff, float *bottom_trans_diff,\n    const float *bottom_data,\n    const float *bottom_rois,\n    const float *bottom_trans,\n    const int no_trans,\n    const float trans_std,\n    const int sample_per_part,\n    const int group_size,\n    const int part_size,\n    const int num_classes,\n    const int channels_each_class)\n{\n  CUDA_KERNEL_LOOP(index, count)\n  {\n    // The output is in order (n, ctop, ph, pw)\n    int pw = index % pooled_width;\n    int ph = (index / pooled_width) % pooled_height;\n    int ctop = (index / pooled_width / pooled_height) % output_dim;\n    int n = index / pooled_width / pooled_height / output_dim;\n    const float *offset_bottom_rois = bottom_rois + n * 5;\n    int roi_batch_ind = offset_bottom_rois[0];\n    float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;\n    float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;\n    float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;\n    float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;\n    // Force too small ROIs to be 1x1\n    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0\n    float roi_height = max(roi_end_h - roi_start_h, 0.1);\n    // Compute w and h at bottom\n    float bin_size_h = roi_height / static_cast<float>(pooled_height);\n    float bin_size_w = roi_width / static_cast<float>(pooled_width);\n    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);\n    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);\n    int part_h = floor(static_cast<float>(ph) / pooled_height * part_size);\n    int part_w = floor(static_cast<float>(pw) / pooled_width * part_size);\n    int class_id = ctop / channels_each_class;\n    float trans_x = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;\n    float trans_y = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;\n    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;\n    wstart += trans_x * roi_width;\n    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;\n    hstart += trans_y * roi_height;\n    if (top_count[index] <= 0)\n    {\n      continue;\n    }\n    float diff_val = top_diff[index] / top_count[index];\n    const float *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;\n    float *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;\n    int gw = floor(static_cast<float>(pw) * group_size / pooled_width);\n    int gh = floor(static_cast<float>(ph) * group_size / pooled_height);\n    gw = min(max(gw, 0), group_size - 1);\n    gh = min(max(gh, 0), group_size - 1);\n    for (int ih = 0; ih < sample_per_part; ih++)\n    {\n      for (int iw = 0; iw < sample_per_part; iw++)\n      {\n        float w = wstart + iw * sub_bin_size_w;\n        float h = hstart + ih * sub_bin_size_h;\n        // bilinear interpolation\n        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)\n        {\n          continue;\n        }\n        w = min(max(w, 0.), width - 1.);\n        h = min(max(h, 0.), height - 1.);\n        int c = (ctop * group_size + gh) * group_size + gw;\n        // backward on feature\n        int x0 = floor(w);\n        int x1 = ceil(w);\n        int y0 = floor(h);\n        int y1 = ceil(h);\n        float dist_x = w - x0, dist_y = h - y0;\n        float q00 = (1 - dist_x) * (1 - dist_y);\n        float q01 = (1 - dist_x) * dist_y;\n        float q10 = dist_x * (1 - dist_y);\n        float q11 = dist_x * dist_y;\n        int bottom_index_base = c * height * width;\n        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);\n        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);\n        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);\n        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);\n        if (no_trans)\n        {\n          continue;\n        }\n        float U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];\n        float U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];\n        float U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];\n        float U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];\n        float diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;\n        diff_x *= roi_width;\n        float diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;\n        diff_y *= roi_height;\n        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);\n        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);\n      }\n    }\n  }\n}\n    ', cuda_src=f"""
    const int no_trans = {no_trans};
    const float spatial_scale = {spatial_scale};
    const int output_dim = {output_dim};
    const int group_size = {group_size};
    const int pooled_size = {pooled_size};
    const int part_size = {part_size};
    const int sample_per_part = {sample_per_part};
    const float trans_std = {trans_std};
    """ + '\n    @alias(out_grad,in0)\n    @alias(input,in1)\n    @alias(bbox,in2)\n    @alias(trans,in3)\n    @alias(top_count,in4)\n    @alias(input_grad,out0)\n    @alias(trans_grad,out1)\n    const int batch = input_shape0;\n  const int channels = input_shape1;\n  const int height = input_shape2;\n  const int width = input_shape3;\n  const int channels_trans = no_trans ? 2 : trans_shape1;\n  const int num_bbox = bbox_shape0;\n  auto pooled_height = pooled_size;\n  auto pooled_width = pooled_size;\n  long out_size = num_bbox * output_dim * pooled_height * pooled_width;\n  const int num_classes = no_trans ? 1 : channels_trans / 2;\n  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;\n  long tmp = out_size % 512L==0? out_size/512L :out_size/512L+1L;\n  dim3 grid(std::min(tmp, 4096L));\n  dim3 block(512);\n  DeformablePSROIPoolBackwardAccKernel<<<grid, block>>>(\n        out_size,\n        out_grad_p,\n        top_count_p,\n        num_bbox,\n        spatial_scale,\n        channels,\n        height,\n        width,\n        pooled_height,\n        pooled_width,\n        output_dim,\n        input_grad_p,\n        trans_grad_p,\n        input_p,\n        bbox_p,\n        trans_p,\n        no_trans,\n        trans_std,\n        sample_per_part,\n        group_size,\n        part_size,\n        num_classes,\n        channels_each_class);\n    ')
    return input_grad, trans_grad


def dcn_v2_pooling_forward(input, bbox, trans, spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std):
    channels = input.shape[1]
    num_bbox = bbox.shape[0]
    assert channels == output_dim, 'input channels and output channels must equal'
    pooled_height = pooled_size
    pooled_width = pooled_size
    output_shape = [(num_bbox, output_dim, pooled_height, pooled_width), (num_bbox, output_dim, pooled_height, pooled_width)]
    output_dtypes = [input.dtype, input.dtype]
    inputs = [input, bbox, trans]
    out, top_count = jt.code(output_shape, output_dtypes, inputs, cuda_header='\n#include<cstdio>\n#include<cstring>\n#include<algorithm>\nusing namespace std;\n#define CUDA_KERNEL_LOOP(i, n)                        \\\n  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \\\n       i < (n);                                       \\\n       i += blockDim.x * gridDim.x)\nconst int CUDA_NUM_THREADS = 1024;\ninline int GET_BLOCKS(const int N)\n{\n  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;\n}\n__device__ float bilinear_interp(\n    const float *data,\n    const float x,\n    const float y,\n    const int width,\n    const int height)\n{\n  int x1 = floor(x);\n  int x2 = ceil(x);\n  int y1 = floor(y);\n  int y2 = ceil(y);\n  float dist_x = static_cast<float>(x - x1);\n  float dist_y = static_cast<float>(y - y1);\n  float value11 = data[y1 * width + x1];\n  float value12 = data[y2 * width + x1];\n  float value21 = data[y1 * width + x2];\n  float value22 = data[y2 * width + x2];\n  float value = (1 - dist_x) * (1 - dist_y) * value11 +\n            (1 - dist_x) * dist_y * value12 +\n            dist_x * (1 - dist_y) * value21 +\n            dist_x * dist_y * value22;\n  return value;\n}\n__global__ void DeformablePSROIPoolForwardKernel(\n    const int count,\n    const float *bottom_data,\n    const float spatial_scale,\n    const int channels,\n    const int height, const int width,\n    const int pooled_height, const int pooled_width,\n    const float *bottom_rois, const float *bottom_trans,\n    const int no_trans,\n    const float trans_std,\n    const int sample_per_part,\n    const int output_dim,\n    const int group_size,\n    const int part_size,\n    const int num_classes,\n    const int channels_each_class,\n    float *top_data,\n    float *top_count)\n{\n  CUDA_KERNEL_LOOP(index, count)\n  {\n    // The output is in order (n, ctop, ph, pw)\n    int pw = index % pooled_width;\n    int ph = (index / pooled_width) % pooled_height;\n    int ctop = (index / pooled_width / pooled_height) % output_dim;\n    int n = index / pooled_width / pooled_height / output_dim;\n    const float *offset_bottom_rois = bottom_rois + n * 5;\n    int roi_batch_ind = offset_bottom_rois[0];\n    float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;\n    float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;\n    float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;\n    float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;\n    // Force too small ROIs to be 1x1\n    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0\n    float roi_height = max(roi_end_h - roi_start_h, 0.1);\n    // Compute w and h at bottom\n    float bin_size_h = roi_height / static_cast<float>(pooled_height);\n    float bin_size_w = roi_width / static_cast<float>(pooled_width);\n    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);\n    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);\n    int part_h = floor(static_cast<float>(ph) / pooled_height * part_size);\n    int part_w = floor(static_cast<float>(pw) / pooled_width * part_size);\n    int class_id = ctop / channels_each_class;\n    float trans_x = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;\n    float trans_y = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;\n    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;\n    wstart += trans_x * roi_width;\n    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;\n    hstart += trans_y * roi_height;\n    float sum = 0;\n    int count = 0;\n    int gw = floor(static_cast<float>(pw) * group_size / pooled_width);\n    int gh = floor(static_cast<float>(ph) * group_size / pooled_height);\n    gw = min(max(gw, 0), group_size - 1);\n    gh = min(max(gh, 0), group_size - 1);\n    const float *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;\n    for (int ih = 0; ih < sample_per_part; ih++)\n    {\n      for (int iw = 0; iw < sample_per_part; iw++)\n      {\n        float w = wstart + iw * sub_bin_size_w;\n        float h = hstart + ih * sub_bin_size_h;\n        // bilinear interpolation\n        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)\n        {\n          continue;\n        }\n        w = min(max(w, 0.), width - 1.);\n        h = min(max(h, 0.), height - 1.);\n        int c = (ctop * group_size + gh) * group_size + gw;\n        float val = bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);\n        sum += val;\n        count++;\n      }\n    }\n    top_data[index] = count == 0 ? static_cast<float>(0) : sum / count;\n    top_count[index] = count;\n  }\n}\n    ', cuda_src=f"""
    const int no_trans = {no_trans};
    const float spatial_scale = {spatial_scale};
    const int output_dim = {output_dim};
    const int group_size = {group_size};
    const int pooled_size = {pooled_size};
    const int part_size = {part_size};
    const int sample_per_part = {sample_per_part};
    const float trans_std = {trans_std};
    """ + '\n    @alias(input,in0)\n    @alias(bbox,in1)\n    @alias(trans,in2)\n    @alias(top_count,out1)\n  const int batch = input_shape0;\n  const int channels = input_shape1;\n  const int height = input_shape2;\n  const int width = input_shape3;\n  const int channels_trans = no_trans ? 2 : trans_shape1;\n  const int num_bbox = bbox_shape0;\n  auto pooled_height = pooled_size;\n  auto pooled_width = pooled_size;\n  long out_size = num_bbox * output_dim * pooled_height * pooled_width;\n  const int num_classes = no_trans ? 1 : channels_trans / 2;\n  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;\n  long tmp = out_size % 512L==0? out_size/512L :out_size/512L+1L;\n  dim3 grid(std::min(tmp, 4096L));\n  dim3 block(512);\n  DeformablePSROIPoolForwardKernel<<<grid, block>>>(\n        out_size,\n        input_p,\n        spatial_scale,\n        channels,\n        height, width,\n        pooled_height,\n        pooled_width,\n        bbox_p,\n        trans_p,\n        no_trans,\n        trans_std,\n        sample_per_part,\n        output_dim,\n        group_size,\n        part_size,\n        num_classes,\n        channels_each_class,\n        out_p,\n        top_count_p);\n    ')
    return out, top_count


class DCNv2Pooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def execute(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            o_shape = (0,) + input.shape[1:]
            offset = jt.empty(o_shape)
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_dim = deform_fc_dim
        if not no_trans:
            self.offset_mask_fc = nn.Sequential(nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim), nn.ReLU(), nn.Linear(self.deform_fc_dim, self.deform_fc_dim), nn.ReLU(), nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3))
            nn.init.constant_(self.offset_mask_fc[4].weight, 0.0)
            nn.init.constant_(self.offset_mask_fc[4].bias, 0.0)

    def execute(self, input, rois):
        o_shape = (0,) + input.shape[1:]
        offset = jt.empty(o_shape)
        if not self.no_trans:
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = jt.chunk(offset_mask, 3, dim=1)
            offset = jt.contrib.concat((o1, o2), dim=1)
            mask = jt.sigmoid(mask)
            return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class ParallelSum(nn.Module):

    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))


class ParallelWeightedSum(nn.Module):

    def __init__(self, sa, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)
        self.split_attention = sa

    def forward(self, x):
        x1 = self.fns[0](x)
        x2 = self.fns[1](x)
        x3 = self.fns[2](x)
        x_all = torch.stack([x1, x2, x3], 1)
        return self.split_attention(x_all)


class Rearrange1(nn.Module):
    """
    'b h w (c s) -> b w c (h s)'
    """

    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def execute(self, x):
        b, h, w, cs = x.shape
        c = cs // self.segments
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, w, c, -1)
        return x


class Rearrange2(nn.Module):
    """
    'b w c (h s) -> b h w (c s)'
    """

    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def execute(self, x):
        b, w, c, hs = x.shape
        h = hs // self.segments
        x = x.reshape(b, w, h, -1)
        x = x.permute(0, 2, 1, 3)
        return x


class Rearrange3(nn.Module):
    """
    'b h w (c s) -> b h c (w s)'
    """

    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def execute(self, x):
        b, h, w, cs = x.shape
        c = cs // self.segments
        x = x.reshape(b, h, c, -1)
        return x


class Rearrange4(nn.Module):
    """
    'b h c (w s) -> b h w (c s)'
    """

    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def execute(self, x):
        b, h, c, ws = x.shape
        w = ws // self.segments
        x = x.view(b, h, w, -1)
        return x


class WeightedPermutator(nn.Module):

    def __init__(self, height, width, d_model, depth, segments, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, nn.Sequential(ParallelWeightedSum(SplitAttention(d_model, k=3), nn.Sequential(Rearrange('b h w (c s) -> b w c (h s)', s=segments), nn.Linear(height * segments, height * segments), Rearrange('b w c (h s) -> b h w (c s)', s=segments)), nn.Sequential(Rearrange('b h w (c s) -> b h c (w s)', s=segments), nn.Linear(width * segments, width * segments), Rearrange('b h c (w s) -> b h w (c s)', s=segments)), nn.Linear(d_model, d_model)), nn.Linear(d_model, d_model))), PreNormResidual(d_model, nn.Sequential(nn.Linear(d_model, d_model * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * expansion_factor, d_model), nn.Dropout(dropout)))) for _ in range(depth)])

    def forward(self, x):
        return self.model(x)


class Permutator(nn.Module):

    def __init__(self, height, width, d_model, depth, segments, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(PreNormResidual(d_model, nn.Sequential(ParallelSum(nn.Sequential(Rearrange('b h w (c s) -> b w c (h s)', s=segments), nn.Linear(height * segments, height * segments), Rearrange('b w c (h s) -> b h w (c s)', s=segments)), nn.Sequential(Rearrange('b h w (c s) -> b h c (w s)', s=segments), nn.Linear(width * segments, width * segments), Rearrange('b h c (w s) -> b h w (c s)', s=segments)), nn.Linear(d_model, d_model)), nn.Linear(d_model, d_model))), PreNormResidual(d_model, nn.Sequential(nn.Linear(d_model, d_model * expansion_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * expansion_factor, d_model), nn.Dropout(dropout)))) for _ in range(depth)])

    def forward(self, x):
        return self.model(x)


class ViP(nn.Module):

    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000, d_model=256, depth=30, segments=14, expansion_factor=4, weighted=True):
        image_size = pair(image_size)
        patch_size = pair(patch_size)
        assert image_size[0] % patch_size[0] == 0, 'image must be divisible by patch size'
        assert image_size[1] % patch_size[1] == 0, 'image must be divisible by patch size'
        assert d_model % segments == 0, 'dimension must be divisible by the number of segments'
        height = image_size[0] // patch_size[0]
        width = image_size[1] // patch_size[1]
        super().__init__()
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size))
        if weighted:
            self.blocks = WeightedPermutator(height, width, d_model, depth, segments, expansion_factor, dropout=0.0)
        else:
            self.blocks = Permutator(height, width, d_model, depth, segments, expansion_factor, dropout=0.0)
        self.mlp_head = nn.Sequential(nn.LayerNorm(d_model), Reduce('b h w c -> b c', 'mean'), nn.Linear(d_model, num_classes))

    def forward(self, x):
        patches = self.patcher(x)
        patches = patches.permute(0, 2, 3, 1)
        embedding = self.blocks(patches)
        out = self.mlp_head(embedding)
        return out


wavemlp_settings = {'T': [[2, 2, 4, 2], [4, 4, 4, 4]], 'S': [[2, 3, 10, 3], [4, 4, 4, 4]], 'M': [[3, 4, 18, 3], [8, 8, 4, 4]]}


class WaveMLP(nn.Module):

    def __init__(self, model_name: str='T', pretrained: str=None, num_classes: int=1000, *args, **kwargs) ->None:
        super().__init__()
        assert model_name in wavemlp_settings.keys(), f'WaveMLP model name should be in {list(wavemlp_settings.keys())}'
        layers, mlp_ratios = wavemlp_settings[model_name]
        embed_dims = [64, 128, 320, 512]
        self.patch_embed = PatchEmbedOverlap(7, 4, 2, embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = nn.Sequential(*[Block(embed_dims[i], mlp_ratios[i]) for _ in range(layers[i])])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i + 1]))
        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.out_indices = [0, 2, 4, 6]
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str=None) ->None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'])
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def return_features(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f'norm{i}')(x)
                outs.append(out)
        return outs

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for blk in self.network:
            x = blk(x)
        x = self.norm(x)
        x = self.head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
        return x


class ATMOp(nn.Module):

    def __init__(self, in_chans, out_chans, stride: int=1, padding: int=0, dilation: int=1, bias: bool=True, dimension: str=''):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension
        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        ATM along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f'{self.dimension} dimension not implemented')
        return deform_conv2d(input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def extra_repr(self) ->str:
        s = self.__class__.__name__ + '('
        s += 'dimension={dimension}'
        s += ', in_chans={in_chans}'
        s += ', out_chans={out_chans}'
        s += ', stride={stride}'
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class ATMLayer(nn.Module):

    def __init__(self, dim, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_h = ATMOp(dim, dim, dimension='h')
        self.atm_w = ATMOp(dim, dim, dimension='w')
        self.fusion = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f'offset shape not match, got {offset.shape}'
        w = self.atm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :]).permute(0, 2, 3, 1)
        h = self.atm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :]).permute(0, 2, 3, 1)
        c = self.atm_c(x)
        a = (w + h + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = w * a[0] + h * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        s = self.__class__.__name__ + ' ('
        s += 'dim: {dim}'
        s += ')'
        return s.format(**self.__dict__)


class ActiveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, share_dim=1, downsample=None, new_offset=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.atm = ATMLayer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.downsample = downsample
        self.new_offset = new_offset
        self.share_dim = share_dim
        if new_offset:
            self.offset_layer = nn.Sequential(norm_layer(dim), nn.Linear(dim, dim * 2 // self.share_dim))
        else:
            self.offset_layer = None

    def forward(self, x, offset=None):
        """
        :param x: [B, H, W, C]
        :param offset: [B, 2C, H, W]
        """
        if self.offset_layer and offset is None:
            offset = self.offset_layer(x).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.atm(self.norm1(x), offset))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.downsample is not None:
            x = self.downsample(x)
        if self.offset_layer:
            return x, offset
        else:
            return x

    def extra_repr(self) ->str:
        s = self.__class__.__name__ + ' ('
        s += 'new_offset: {offset}'
        s += ', share_dim: {share_dim}'
        s += ')'
        return s.format(**self.__dict__)


class PEG(nn.Module):
    """
    PEG
    from https://arxiv.org/abs/2102.10882
    """

    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dim)
        self.stride = stride

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x_conv = x
        x_conv = x_conv.permute(0, 3, 1, 2)
        if self.stride == 1:
            x = self.proj(x_conv) + x_conv
        else:
            x = self.proj(x_conv)
        x = x.permute(0, 2, 3, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlaped patch embedding, implemeted with 2D conv
    """

    def __init__(self, patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = self.proj(x)
        return x


class ActiveMLP(nn.Module):
    """
    ActiveMLP
    https://arxiv.org/abs/2203.06108
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], embed_dims=[64, 128, 320, 512], mlp_ratios=[4, 4, 4, 4], share_dims=[1, 1, 1, 1], drop_path_rate=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, intv=2, **kwargs):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.intv = intv
        None
        None
        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        ii = 0
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            _block = nn.ModuleList([ActiveBlock(embed_dims[i], mlp_ratio=mlp_ratios[i], drop_path=dpr[ii + j], share_dim=share_dims[i], act_layer=act_layer, norm_layer=norm_layer, downsample=Downsample(embed_dims[i], embed_dims[i + 1]) if i < len(depths) - 1 and j == depths[i] - 1 else None, new_offset=j % self.intv == 0 and j != depths[i] - 1) for j in range(depths[i])])
            self.blocks.append(_block)
            ii += depths[i]
        self.pos_blocks = nn.ModuleList([PEG(ed, ed) for ed in embed_dims])
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, ATMOp):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return set([('pos_blocks.' + n) for n, p in self.pos_blocks.named_parameters()])

    def forward_blocks(self, x):
        for i in range(len(self.depths)):
            for j, blk in enumerate(self.blocks[i]):
                if j % self.intv == 0 and j != len(self.blocks[i]) - 1:
                    x = self.pos_blocks[i](x)
                    x, offset = blk(x)
                else:
                    x = blk(x, offset)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = self.forward_blocks(x)
        x = self.norm(x)
        y = self.head(x.mean(1))
        return y


class GlobalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class BlockLayerScale(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-05):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class GFNet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, mlp_ratio=4.0, representation_size=None, uniform_drop=False, drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h = img_size // patch_size
        w = h // 2 + 1
        if uniform_drop:
            None
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            None
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if dropcls > 0:
            None
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


class GFNetPyramid(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, embed_dim=[64, 128, 256, 512], depth=[2, 2, 10, 4], mlp_ratio=[4, 4, 4, 4], drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = nn.ModuleList()
        patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0])
        num_patches = patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.patch_embed.append(patch_embed)
        sizes = [56, 28, 14, 7]
        for i in range(4):
            sizes[i] = sizes[i] * img_size // 224
        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1
            if no_layerscale:
                None
                blk = nn.Sequential(*[Block(dim=embed_dim[i], mlp_ratio=mlp_ratio[i], drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w) for j in range(depth[i])])
            else:
                None
                blk = nn.Sequential(*[BlockLayerScale(dim=embed_dim[i], mlp_ratio=mlp_ratio[i], drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w, init_values=init_values) for j in range(depth[i])])
            self.blocks.append(blk)
            cur += depth[i]
        self.norm = norm_layer(embed_dim[-1])
        self.head = nn.Linear(self.num_features, num_classes)
        if dropcls > 0:
            None
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)
        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Aff,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiLSTM2D,
     lambda: ([], {'d_model': 4, 'hidden_d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvDownsample,
     lambda: ([], {'embedding_dim_in': 4, 'embedding_dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvMixer,
     lambda: ([], {'dim': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ConvStage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ConvTokenizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CrossRegion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FFNBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalPerceptron,
     lambda: ([], {'input_channels': 4, 'internal_neurons': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Level,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPblock,
     lambda: ([], {'num_patch': 4, 'dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PATM,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ParallelSum,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchEmbedOverlap,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PatchEmbedding,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreNormResidual,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RepMLPBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'h': 4, 'w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResMLP,
     lambda: ([], {'num_patch': 4, 'd_model': 4, 'depth': 1, 'expansion_factor': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (S2Block,
     lambda: ([], {'d_model': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sequencer2DBlock,
     lambda: ([], {'d_model': 4, 'depth': 1, 'hidden_d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Spatial_Shift,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WaveMLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (gMLPBlock,
     lambda: ([], {'d_model': 4, 'd_ffn': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_liuruiyang98_Jittor_MLP(_paritybench_base):
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

