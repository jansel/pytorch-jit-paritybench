import sys
_module = sys.modules[__name__]
del sys
calc_metrics = _module
dataset = _module
make_dataset_from_frames = _module
make_dataset_from_videos = _module
make_dataset_from_youtube = _module
utils = _module
dnnlib = _module
util = _module
generate = _module
frechet_video_distance = _module
video_inception_score = _module
ada_augment = _module
diff_augment = _module
generator_lres = _module
generator_sres = _module
video_gan_lres = _module
video_gan_sres = _module
plot_color_similarity = _module
torch_utils = _module
custom_ops = _module
distributed = _module
misc = _module
ops = _module
bias_act = _module
conv2d_gradfix = _module
conv2d_resample = _module
filtered_lrelu = _module
fma = _module
grid_sample_gradfix = _module
upfirdn2d = _module
persistence = _module
training_stats = _module
train_lres = _module
train_sres = _module
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


from typing import Any


from typing import Optional


import numpy as np


import torch


from torch.utils.data import Dataset


import scipy.signal


import torch.nn.functional as F


import math


from collections.abc import Iterator


from typing import Union


import torch.distributed as dist


import torch.nn as nn


import scipy.optimize


import copy


from torch.optim import Adam


import matplotlib.pyplot as plt


import re


import uuid


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


import warnings


import inspect


import types


import time


import functools


from collections.abc import Iterable


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [(x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device)) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def rotate2d(theta, **kwargs):
    return matrix([torch.cos(theta), torch.sin(-theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1], **kwargs)


def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    return matrix([vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0], [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0], [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0], [0, 0, 0, 1], **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix([sx, 0, 0], [0, sy, 0], [0, 0, 1], **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


def scale3d(sx, sy, sz, **kwargs):
    return matrix([sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1], **kwargs)


def translate2d(tx, ty, **kwargs):
    return matrix([1, 0, tx], [0, 1, ty], [0, 0, 1], **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def translate3d(tx, ty, tz, **kwargs):
    return matrix([1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1], **kwargs)


def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(module_name='upfirdn2d_plugin', sources=['upfirdn2d.cpp', 'upfirdn2d.cu'], headers=['upfirdn2d.h'], source_dir=os.path.dirname(__file__), extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'])
    return True


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
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


_plugin = None


_upfirdn2d_cuda_cache = dict()


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
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
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
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


wavelets = {'haar': [0.7071067811865476, 0.7071067811865476], 'db1': [0.7071067811865476, 0.7071067811865476], 'db2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'db3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'db4': [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523], 'db5': [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125], 'db6': [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017], 'db7': [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236], 'db8': [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161], 'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427], 'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728], 'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148], 'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255], 'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609]}


def upsample2d_wrapper(x, f, up=2, **kwargs) ->torch.Tensor:
    upx, upy = upfirdn2d._parse_scaling(up)
    output_size = x.numel() * upx * upy
    int_max = 2 ** 31 - 1
    if output_size < int_max:
        return upfirdn2d.upsample2d(x, f, up=up, **kwargs)
    split_dim = np.argmax(x.shape[:2])
    num_chunks = math.ceil(output_size / (int_max - 1))
    assert num_chunks <= x.size(split_dim), 'Tensor is too large'
    chunks = []
    for chunk in x.chunk(num_chunks, dim=split_dim):
        chunk = upfirdn2d.upsample2d(chunk, f, up=up, **kwargs)
        chunks.append(chunk)
    output = torch.cat(chunks, dim=split_dim)
    return output


_bias_act_cuda_cache = dict()


_null_tensor = torch.empty([0])


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
            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride(1) == 1 else torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not _null_tensor:
                y = _plugin.bias_act(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, spec.cuda_idx, alpha, gain, clamp)
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
            ctx.memory_format = torch.channels_last if dy.ndim > 2 and dy.stride(1) == 1 else torch.contiguous_format
            dx = _plugin.bias_act(dy, b, x, y, _null_tensor, 1, dim, spec.cuda_idx, alpha, gain, clamp)
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
                d_x = _plugin.bias_act(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx, alpha, gain, clamp)
            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])
            return d_dy, d_x, d_b, d_y
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    """Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)


def normalize_2nd_moment(tensor: torch.Tensor, dim: Union[int, tuple[int, ...]]=1, eps: float=1e-08) ->torch.Tensor:
    return tensor * tensor.square().mean(dim=dim, keepdim=True).add(eps).rsqrt()


def bias_act_wrapper(x, b=None, dim=1, **kwargs) ->torch.Tensor:
    int_max = 2 ** 31 - 1
    if x.numel() < int_max:
        return bias_act.bias_act(x, b=b, dim=dim, **kwargs)
    split_dim = np.argmax(x.size())
    num_chunks = math.ceil(x.numel() / (int_max - 1))
    assert num_chunks <= x.size(split_dim), 'Tensor is too large'
    if split_dim == dim and b is not None:
        b_chunks = b.chunk(num_chunks, dim=0)
    else:
        b_chunks = [b] * num_chunks
    chunks = []
    for x_chunk, b_chunk in zip(x.chunk(num_chunks, dim=split_dim), b_chunks):
        chunk = bias_act.bias_act(x_chunk, b=b_chunk, dim=dim, **kwargs)
        chunks.append(chunk)
    output = torch.cat(chunks, dim=split_dim)
    return output


def center_crop(input: torch.Tensor, width: Optional[int]=None, height: Optional[int]=None, seq_length: Optional[int]=None) ->torch.Tensor:
    input_dim = input.dim()
    assert input_dim in (3, 5)
    if width is not None:
        assert input_dim == 5
        x0 = (input.size(4) - width) // 2
        x1 = x0 + width
        input = input[:, :, :, :, x0:x1]
    if height is not None:
        assert input_dim == 5
        y0 = (input.size(3) - height) // 2
        y1 = y0 + height
        input = input[:, :, :, y0:y1]
    if seq_length is not None:
        t0 = (input.size(2) - seq_length) // 2
        t1 = t0 + seq_length
        input = input[:, :, t0:t1]
    return input


def temporal_modulated_conv3d(input: torch.Tensor, weight: torch.Tensor, style: torch.Tensor, input_gain: Optional[torch.Tensor]=None, padding: tuple[int, int, int]=(0, 0, 0), demodulate: bool=True) ->torch.Tensor:
    assert input.dim() == 5
    batch_size, in_channels = input.size(0), input.size(1)
    misc.assert_shape(weight, (None, in_channels, None, None, None))
    misc.assert_shape(style, (batch_size, in_channels, None))
    if demodulate:
        weight = weight / weight.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
        style = style / style.abs().amax(dim=(1, 2), keepdim=True)
    num_inputs = np.prod(weight.shape[1:])
    weight = weight / math.sqrt(num_inputs)
    if demodulate:
        demodulation = torch.einsum('oizyx,nit->not', weight.square(), style.square())
        demodulation = demodulation.add(1e-08).rsqrt()
        demodulation = einops.rearrange(demodulation, 'n co t -> n co t 1 1')
    if input_gain is not None:
        assert input_gain.dim() == 0
        input = input * input_gain
    style = einops.rearrange(style, 'n ci t -> n ci t 1 1')
    input = input * style.type(input.dtype)
    output = F.conv3d(input, weight.type(input.dtype), padding=padding)
    if demodulate:
        output = output * demodulation.type(output.dtype)
    return output


_filtered_lrelu_cuda_cache = dict()


def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or clamp == float(clamp) and clamp >= 0
    clamp = float(clamp if clamp is not None else 'inf')
    key = up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]


    class FilteredLReluCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]
            if si is None:
                si = torch.empty([0])
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)
            write_signs = si.numel() == 0 and (x.requires_grad or b.requires_grad)
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn('low-performance memory layout detected in filtered_lrelu input', RuntimeWarning)
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                    warnings.warn('filtered_lrelu called with non-default cuda stream but concurrent execution is not supported', RuntimeWarning)
                y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1
            if return_code < 0:
                warnings.warn('filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback', RuntimeWarning)
                y = x.add(b.unsqueeze(-1).unsqueeze(-1))
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs)
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter)
            ctx.save_for_backward(fu, fd, si if si.numel() else so)
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = None
            dfu = None
            assert not ctx.needs_input_grad[1]
            dfd = None
            assert not ctx.needs_input_grad[2]
            db = None
            dsi = None
            assert not ctx.needs_input_grad[4]
            dsx = None
            assert not ctx.needs_input_grad[5]
            dsy = None
            assert not ctx.needs_input_grad[6]
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [fu.shape[-1] - 1 + (fd.shape[-1] - 1) - px0, xw * up - yw * down + px0 - (up - 1), fu.shape[0] - 1 + (fd.shape[0] - 1) - py0, xh * up - yh * down + py0 - (up - 1)]
                gg = gain * up ** 2 / down ** 2
                ff = not flip_filter
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)
            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])
            return dx, dfu, dfd, db, dsi, dsx, dsy
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda


def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    """Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)

