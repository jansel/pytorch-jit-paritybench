import sys
_module = sys.modules[__name__]
del sys
stylegan_demo = _module
stylegan_ffhq1024 = _module
stylegan_ffhq1024_val = _module
stylegan_ffhq256 = _module
stylegan_ffhq256_encoder_y = _module
stylegan_ffhq256_val = _module
convert_model = _module
converters = _module
pggan_converter = _module
config = _module
dataset = _module
dataset_tool = _module
legacy = _module
loss = _module
metrics = _module
frechet_inception_distance = _module
inception_score = _module
ms_ssim = _module
sliced_wasserstein = _module
misc = _module
networks = _module
tfutil = _module
train = _module
util_scripts = _module
stylegan2_converter = _module
dnnlib = _module
submission = _module
internal = _module
local = _module
run_context = _module
submit = _module
tflib = _module
autosummary = _module
custom_ops = _module
network = _module
ops = _module
fused_bias_act = _module
upfirdn_2d = _module
optimizer = _module
util = _module
linear_separability = _module
metric_base = _module
metric_defaults = _module
perceptual_path_length = _module
precision_recall = _module
pretrained_networks = _module
projector = _module
run_generator = _module
run_metrics = _module
run_projector = _module
run_training = _module
training = _module
networks_stylegan = _module
networks_stylegan2 = _module
training_loop = _module
stylegan2ada_pth_converter = _module
calc_metrics = _module
generate = _module
legacy = _module
kernel_inception_distance = _module
metric_main = _module
metric_utils = _module
perceptual_path_length = _module
precision_recall = _module
projector = _module
style_mixing = _module
torch_utils = _module
custom_ops = _module
misc = _module
bias_act = _module
conv2d_gradfix = _module
conv2d_resample = _module
fma = _module
grid_sample_gradfix = _module
upfirdn2d = _module
persistence = _module
training_stats = _module
train = _module
augment = _module
dataset = _module
loss = _module
networks = _module
training_loop = _module
stylegan2ada_tf_converter = _module
stylegan_converter = _module
run = _module
generate_figures = _module
pretrained_example = _module
networks_progan = _module
datasets = _module
dataloaders = _module
datasets = _module
distributed_sampler = _module
transforms = _module
fid = _module
inception = _module
models = _module
encoder = _module
model_zoo = _module
perceptual_model = _module
pggan_discriminator = _module
pggan_generator = _module
stylegan2_discriminator = _module
stylegan2_generator = _module
stylegan_discriminator = _module
stylegan_generator = _module
sync_op = _module
runners = _module
base_encoder_runner = _module
base_gan_runner = _module
base_runner = _module
controllers = _module
base_controller = _module
cache_cleaner = _module
checkpointer = _module
fid_evaluator = _module
lr_scheduler = _module
progress_scheduler = _module
running_logger = _module
snapshoter = _module
timer = _module
encoder_runner = _module
losses = _module
encoder_loss = _module
logistic_gan_loss = _module
optimizer = _module
running_stats = _module
stylegan_runner = _module
synthesize = _module
test = _module
train = _module
utils = _module
logger = _module
logger_test = _module
misc = _module
visualizer = _module

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


import warnings


import numpy as np


import tensorflow as tf


import torch


import re


import copy


from typing import List


from typing import Optional


import time


import uuid


from time import perf_counter


import torch.nn.functional as F


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


import inspect


import types


import scipy.signal


from torch.utils.data import DataLoader


import string


from torch.utils.data import Dataset


import math


from typing import TypeVar


from typing import Iterator


from torch.utils.data import Sampler


import torch.distributed as dist


import torch.nn as nn


from collections import namedtuple


from torch.jit.annotations import Optional


from torch import Tensor


from collections import OrderedDict


from copy import deepcopy


from torch.optim import lr_scheduler


from torch.utils.tensorboard import SummaryWriter


import random


import torch.multiprocessing as mp


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


class PPLSampler(torch.nn.Module):

    def __init__(self, G, G_kwargs, epsilon, space, sampling, crop, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self, c):
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)
        if self.space == 'w':
            w0, w1 = self.G.mapping(z=torch.cat([z0, z1]), c=torch.cat([c, c])).chunk(2)
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else:
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0, wt1 = self.G.mapping(z=torch.cat([zt0, zt1]), c=torch.cat([c, c])).chunk(2)
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))
        img = self.G.synthesis(ws=torch.cat([wt0, wt1]), noise_mode='const', force_fp32=True, **self.G_kwargs)
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c * 3:c * 7, c * 2:c * 6]
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])
        img = (img + 1) * (255 / 2)
        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist


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
    global _inited, _plugin
    if not _inited:
        sources = ['upfirdn2d.cpp', 'upfirdn2d.cu']
        sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
        try:
            _plugin = custom_ops.get_plugin('upfirdn2d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
        except:
            warnings.warn('Failed to build CUDA kernels for upfirdn2d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
    return _plugin is not None


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
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, np.sqrt(gain))
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
            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride()[1] == 1 else torch.contiguous_format
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
            ctx.memory_format = torch.channels_last if dy.ndim > 2 and dy.stride()[1] == 1 else torch.contiguous_format
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


def _get_weight_shape(w):
    with misc.suppress_tracer_warnings():
        shape = [int(sz) for sz in w.shape]
    misc.assert_shape(w, shape)
    return shape


def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.
    """
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    if not flip_weight:
        w = w.flip([2, 3])
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)] and not transpose:
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x
                w = w
                x = conv2d_gradfix.conv2d(x, w, groups=groups)
            return x
    op = conv2d_gradfix.conv_transpose2d if transpose else conv2d_gradfix.conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims + 1:])
    assert x.shape == shape
    return x


class _FusedMultiplyAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, c):
        out = torch.addcmul(c, a, b)
        ctx.save_for_backward(a, b)
        ctx.c_shape = c.shape
        return out

    @staticmethod
    def backward(ctx, dout):
        a, b = ctx.saved_tensors
        c_shape = ctx.c_shape
        da = None
        db = None
        dc = None
        if ctx.needs_input_grad[0]:
            da = _unbroadcast(dout * b, a.shape)
        if ctx.needs_input_grad[1]:
            db = _unbroadcast(dout * a, b.shape)
        if ctx.needs_input_grad[2]:
            dc = _unbroadcast(dout, c_shape)
        return da, db, dc


def fma(a, b, c):
    return _FusedMultiplyAdd.apply(a, b, c)


def resize_image(image, size):
    """Resizes image to target size.

    NOTE: We use adaptive average pooing for image resizing. Instead of bilinear
    interpolation, average pooling is able to acquire information from more
    pixels, such that the resized results can be with higher quality.

    Args:
        image: The input image tensor, with shape [C, H, W], to resize.
        size: An integer or a tuple of integer, indicating the target size.

    Returns:
        An image tensor with target size.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
        ValueError: If the input `image` is not with shape [C, H, W].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [C, H, W], but `{image.shape}` is received!')
    image = F.adaptive_avg_pool2d(image.unsqueeze(0), size).squeeze(0)
    return image


class ImageResizing(nn.Module):
    """Implements the image resizing layer."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image):
        return resize_image(image, self.size)


def normalize_image(image, mean=127.5, std=127.5):
    """Normalizes image by subtracting mean and dividing std.

    Args:
        image: The input image tensor to normalize.
        mean: The mean value to subtract from the input tensor. (default: 127.5)
        std: The standard deviation to normalize the input tensor. (default:
            127.5)

    Returns:
        A normalized image tensor.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, but `{type(image)}` is received!')
    out = (image - mean) / std
    return out


class ImageNormalization(nn.Module):
    """Implements the image normalization layer."""

    def __init__(self, mean=127.5, std=127.5):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        return normalize_image(image, self.mean, self.std)


def normalize_latent_code(latent_code, adjust_norm=True):
    """Normalizes latent code.

    NOTE: The latent code will always be normalized along the last axis.
    Meanwhile, if `adjust_norm` is set as `True`, the norm of the result will be
    adjusted to `sqrt(latent_code.shape[-1])` in order to avoid too small value.

    Args:
        latent_code: The input latent code tensor to normalize.
        adjust_norm: Whether to adjust the norm of the output. (default: True)

    Returns:
        A normalized latent code tensor.

    Raises:
        TypeError: If the input `latent_code` is not with type `torch.Tensor`.
    """
    if not isinstance(latent_code, torch.Tensor):
        raise TypeError(f'Input latent code should be with type `torch.Tensor`, but `{type(latent_code)}` is received!')
    dim = latent_code.shape[-1]
    norm = latent_code.pow(2).sum(-1, keepdim=True).pow(0.5)
    out = latent_code / norm
    if adjust_norm:
        out = out * dim ** 0.5
    return out


class LatentCodeNormalization(nn.Module):
    """Implements the latent code normalization layer."""

    def __init__(self, adjust_norm=True):
        super().__init__()
        self.adjust_norm = adjust_norm

    def forward(self, latent_code):
        return normalize_latent_code(latent_code, self.adjust_norm)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None, align_tf=False):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
        self.pool_include_padding = not align_tf

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None, align_tf=False):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.pool_include_padding = not align_tf

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None, align_tf=False, use_max_pool=False):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.pool_include_padding = not align_tf
        self.use_max_pool = use_max_pool

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        if self.use_max_pool:
            branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, inception_blocks=None, init_weights=True, align_tf=True):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.align_tf = align_tf
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32, align_tf=self.align_tf)
        self.Mixed_5c = inception_a(256, pool_features=64, align_tf=self.align_tf)
        self.Mixed_5d = inception_a(288, pool_features=64, align_tf=self.align_tf)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128, align_tf=self.align_tf)
        self.Mixed_6c = inception_c(768, channels_7x7=160, align_tf=self.align_tf)
        self.Mixed_6d = inception_c(768, channels_7x7=160, align_tf=self.align_tf)
        self.Mixed_6e = inception_c(768, channels_7x7=192, align_tf=self.align_tf)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280, align_tf=self.align_tf)
        self.Mixed_7c = inception_e(2048, use_max_pool=self.align_tf)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x, output_logits=False):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = torch.flatten(x, 1)
        if output_logits:
            x = self.fc(x)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x, output_logits=False):
        x = self._transform_input(x)
        x, aux = self._forward(x, output_logits)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn('Scripted Inception3 always returns Inception3 Tuple')
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class BasicBlock(nn.Module):
    """Implementation of ResNet BasicBlock."""
    expansion = 1

    def __init__(self, inplanes, planes, base_width=64, stride=1, groups=1, dilation=1, norm_layer=None, downsample=None):
        super().__init__()
        if base_width != 64:
            raise ValueError(f'BasicBlock of ResNet only supports `base_width=64`, but {base_width} received!')
        if stride not in [1, 2]:
            raise ValueError(f'BasicBlock of ResNet only supports `stride=1` and `stride=2`, but {stride} received!')
        if groups != 1:
            raise ValueError(f'BasicBlock of ResNet only supports `groups=1`, but {groups} received!')
        if dilation != 1:
            raise ValueError(f'BasicBlock of ResNet only supports `dilation=1`, but {dilation} received!')
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=1, dilation=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    """Implementation of ResNet Bottleneck."""
    expansion = 4

    def __init__(self, inplanes, planes, base_width=64, stride=1, groups=1, dilation=1, norm_layer=None, downsample=None):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'Bottlenet of ResNet only supports `stride=1` and `stride=2`, but {stride} received!')
        width = int(planes * (base_width / 64)) * groups
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=width, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=planes * self.expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + identity)
        return out


class FPN(nn.Module):
    """Implementation of Feature Pyramid Network (FPN).

    The input of this module is a pyramid of features with reducing resolutions.
    Then, this module fuses these multi-level features from `top_level` to
    `bottom_level`. In particular, starting from the `top_level`, each feature
    is convoluted, upsampled, and fused into its previous feature (which is also
    convoluted).

    Args:
        pyramid_channels: A list of integers, each of which indicates the number
            of channels of the feature from a particular level.
        out_channels: Number of channels for each output.

    Returns:
        A list of feature maps, each of which has `out_channels` channels.
    """

    def __init__(self, pyramid_channels, out_channels):
        super().__init__()
        assert isinstance(pyramid_channels, (list, tuple))
        self.num_levels = len(pyramid_channels)
        self.lateral_conv_list = nn.ModuleList()
        self.feature_conv_list = nn.ModuleList()
        for i in range(self.num_levels):
            in_channels = pyramid_channels[i]
            self.lateral_conv_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True))
            self.feature_conv_list.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, inputs):
        if len(inputs) != self.num_levels:
            raise ValueError('Number of inputs and `num_levels` mismatch!')
        laterals = []
        for i in range(self.num_levels):
            laterals.append(self.lateral_conv_list[i](inputs[i]))
        for i in range(self.num_levels - 1, 0, -1):
            scale_factor = laterals[i - 1].shape[2] // laterals[i].shape[2]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], mode='nearest', scale_factor=scale_factor)
        outputs = []
        for i, lateral in enumerate(laterals):
            outputs.append(self.feature_conv_list[i](lateral))
        return outputs


class SAM(nn.Module):
    """Implementation of Spatial Alignment Module (SAM).

    The input of this module is a pyramid of features with reducing resolutions.
    Then this module downsamples all levels of feature to the minimum resolution
    and fuses it with the smallest feature map.

    Args:
        pyramid_channels: A list of integers, each of which indicates the number
            of channels of the feature from a particular level.
        out_channels: Number of channels for each output.

    Returns:
        A list of feature maps, each of which has `out_channels` channels.
    """

    def __init__(self, pyramid_channels, out_channels):
        super().__init__()
        assert isinstance(pyramid_channels, (list, tuple))
        self.num_levels = len(pyramid_channels)
        self.fusion_conv_list = nn.ModuleList()
        for i in range(self.num_levels):
            in_channels = pyramid_channels[i]
            self.fusion_conv_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, inputs):
        if len(inputs) != self.num_levels:
            raise ValueError('Number of inputs and `num_levels` mismatch!')
        output_res = inputs[-1].shape[2:]
        for i in range(self.num_levels - 1, -1, -1):
            if i != self.num_levels - 1:
                inputs[i] = F.adaptive_avg_pool2d(inputs[i], output_res)
            inputs[i] = self.fusion_conv_list[i](inputs[i])
            if i != self.num_levels - 1:
                inputs[i] = inputs[i] + inputs[-1]
        return inputs


class CodeHead(nn.Module):
    """Implementation of the task-head to produce inverted codes."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        if norm_layer is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(out_channels)

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        latent = self.fc(x)
        latent = latent.unsqueeze(2).unsqueeze(3)
        latent = self.norm(latent)
        return latent.flatten(start_dim=1)


_FINAL_RES = 4


_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]


class EncoderNet(nn.Module):
    """Define the ResNet-based encoder network for GAN inversion.

    On top of the backbone, there are several task-heads to produce inverted
    codes. Please use `latent_dim` and `num_latents_per_head` to define the
    structure.

    Settings for the encoder network:

    (1) resolution: The resolution of the output image.
    (2) latent_dim: Dimension of the latent space. A number (one code will be
        produced), or a list of numbers regarding layer-wise latent codes.
    (3) num_latents_per_head: Number of latents that is produced by each head.
    (4) image_channels: Number of channels of the output image. (default: 3)

    ResNet-related settings:

    (1) network_depth: Depth of the network, like 18 for ResNet18. (default: 18)
    (2) inplanes: Number of channels of the first convolutional layer.
        (default: 64)
    (3) groups: Groups of the convolution, used in ResNet. (default: 1)
    (4) width_per_group: Number of channels per group, used in ResNet.
        (default: 64)
    (5) replace_stride_with_dilation: Wether to replace stride with dilation,
        used in ResNet. (default: None)
    (6) norm_layer: Normalization layer used in the encoder.
        (default: nn.BatchNorm2d)
    (7) max_channels: Maximum number of channels in each layer. (default: 512)

    Task-head related settings:

    (1) use_fpn: Whether to use Feature Pyramid Network (FPN) before outputing
        the latent code. (default: True)
    (2) fpn_channels: Number of channels used in FPN. (default: 512)
    (3) use_sam: Whether to use Spatial Alignment Module (SAM) before outputing
        the latent code. (default: True)
    (4) sam_channels: Number of channels used in SAM. (default: 512)
    """
    arch_settings = {(18): (BasicBlock, [2, 2, 2, 2]), (34): (BasicBlock, [3, 4, 6, 3]), (50): (Bottleneck, [3, 4, 6, 3]), (101): (Bottleneck, [3, 4, 23, 3]), (152): (Bottleneck, [3, 8, 36, 3])}

    def __init__(self, resolution, latent_dim, num_latents_per_head, image_channels=3, network_depth=18, inplanes=64, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=nn.BatchNorm2d, max_channels=512, use_fpn=True, fpn_channels=512, use_sam=True, sam_channels=512):
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if network_depth not in self.arch_settings:
            raise ValueError(f'Invalid network depth: `{network_depth}`!\nOptions allowed: {list(self.arch_settings.keys())}.')
        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        assert isinstance(latent_dim, (list, tuple))
        assert isinstance(num_latents_per_head, (list, tuple))
        assert sum(num_latents_per_head) == len(latent_dim)
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.num_latents_per_head = num_latents_per_head
        self.num_heads = len(self.num_latents_per_head)
        self.image_channels = image_channels
        self.inplanes = inplanes
        self.network_depth = network_depth
        self.groups = groups
        self.dilation = 1
        self.base_width = width_per_group
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if norm_layer is None or norm_layer == nn.BatchNorm2d:
            norm_layer = nn.SyncBatchNorm
        self.norm_layer = norm_layer
        self.max_channels = max_channels
        self.use_fpn = use_fpn
        self.fpn_channels = fpn_channels
        self.use_sam = use_sam
        self.sam_channels = sam_channels
        block_fn, num_blocks_per_stage = self.arch_settings[network_depth]
        self.num_stages = int(np.log2(resolution // _FINAL_RES)) - 1
        for i in range(4, self.num_stages):
            num_blocks_per_stage.append(1)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * self.num_stages
        self.conv1 = nn.Conv2d(in_channels=self.image_channels, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_channels = [self.inplanes]
        for i in range(1, self.num_stages + 1):
            channels = min(self.max_channels, self.inplanes * 2 ** (i - 1))
            num_blocks = num_blocks_per_stage[i - 1]
            stride = 1 if i == 1 else 2
            dilate = replace_stride_with_dilation[i - 1]
            self.add_module(f'layer{i}', self._make_stage(block_fn=block_fn, planes=channels, num_blocks=num_blocks, stride=stride, dilate=dilate))
            self.stage_channels.append(channels)
        if self.num_heads > len(self.stage_channels):
            raise ValueError(f'Number of task heads is larger than number of stages! Please reduce the number of heads.')
        if self.num_heads == 1:
            self.use_fpn = False
            self.use_sam = False
        if self.use_fpn:
            fpn_pyramid_channels = self.stage_channels[-self.num_heads:]
            self.fpn = FPN(pyramid_channels=fpn_pyramid_channels, out_channels=self.fpn_channels)
        if self.use_sam:
            if use_fpn:
                sam_pyramid_channels = [self.fpn_channels] * self.num_heads
            else:
                sam_pyramid_channels = self.stage_channels[-self.num_heads:]
            self.sam = SAM(pyramid_channels=sam_pyramid_channels, out_channels=self.sam_channels)
        self.head_list = nn.ModuleList()
        for head_idx in range(self.num_heads):
            if self.use_sam:
                in_channels = self.sam_channels
            elif self.use_fpn:
                in_channels = self.fpn_channels
            else:
                in_channels = self.stage_channels[head_idx - self.num_heads]
            in_channels = in_channels * _FINAL_RES * _FINAL_RES
            start_latent_idx = sum(self.num_latents_per_head[:head_idx])
            end_latent_idx = sum(self.num_latents_per_head[:head_idx + 1])
            out_channels = sum(self.latent_dim[start_latent_idx:end_latent_idx])
            self.head_list.append(CodeHead(in_channels=in_channels, out_channels=out_channels, norm_layer=self.norm_layer))

    def _make_stage(self, block_fn, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block_fn.expansion, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False), norm_layer(planes * block_fn.expansion))
        blocks = []
        blocks.append(block_fn(inplanes=self.inplanes, planes=planes, base_width=self.base_width, stride=stride, groups=self.groups, dilation=previous_dilation, norm_layer=norm_layer, downsample=downsample))
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, num_blocks):
            blocks.append(block_fn(inplanes=self.inplanes, planes=planes, base_width=self.base_width, stride=1, groups=self.groups, dilation=self.dilation, norm_layer=norm_layer, downsample=None))
        return nn.Sequential(*blocks)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = [x]
        for i in range(1, self.num_stages + 1):
            x = getattr(self, f'layer{i}')(x)
            features.append(x)
        features = features[-self.num_heads:]
        if self.use_fpn:
            features = self.fpn(features)
        if self.use_sam:
            features = self.sam(features)
        else:
            final_size = features[-1].shape[2:]
            for i in range(self.num_heads - 1):
                features[i] = F.adaptive_avg_pool2d(features[i], final_size)
        outputs = []
        for head_idx in range(self.num_heads):
            codes = self.head_list[head_idx](features[head_idx])
            start_latent_idx = sum(self.num_latents_per_head[:head_idx])
            end_latent_idx = sum(self.num_latents_per_head[:head_idx + 1])
            split_size = self.latent_dim[start_latent_idx:end_latent_idx]
            outputs.extend(torch.split(codes, split_size, dim=1))
        max_dim = max(self.latent_dim)
        for i, dim in enumerate(self.latent_dim):
            if dim < max_dim:
                outputs[i] = F.pad(outputs[i], (0, max_dim - dim))
            outputs[i] = outputs[i].unsqueeze(1)
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        return self._forward_impl(x)


_MEAN_STATS = 103.939, 116.779, 123.68


class PerceptualModel(nn.Module):
    """Defines the VGG16 structure as the perceptual network.

    This model takes `RGB` images with data format `NCHW` as the raw inputs, and
    outputs the perceptual feature. This following operations will be performed
    to preprocess the inputs to match the preprocessing during the model
    training:
    (1) Shift pixel range to [0, 255].
    (2) Change channel order to `BGR`.
    (3) Subtract the statistical mean.

    NOTE: The three fully connected layers on top of the model are dropped.
    """

    def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0, pretrained_weight_path=None):
        """Defines the network structure.

        Args:
            output_layer_idx: Index of layer whose output will be used as the
                perceptual feature. (default: 23, which is the `block4_conv3`
                layer activated by `ReLU` function)
            min_val: Minimum value of the raw input. (default: -1.0)
            max_val: Maximum value of the raw input. (default: 1.0)
            pretrained_weight_path: Path to the pretrained weights.
                (default: None)
        """
        super().__init__()
        self.vgg16 = nn.Sequential(OrderedDict({'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 'layer1': nn.ReLU(inplace=True), 'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 'layer3': nn.ReLU(inplace=True), 'layer4': nn.MaxPool2d(kernel_size=2, stride=2), 'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 'layer6': nn.ReLU(inplace=True), 'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 'layer8': nn.ReLU(inplace=True), 'layer9': nn.MaxPool2d(kernel_size=2, stride=2), 'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 'layer11': nn.ReLU(inplace=True), 'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 'layer13': nn.ReLU(inplace=True), 'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 'layer15': nn.ReLU(inplace=True), 'layer16': nn.MaxPool2d(kernel_size=2, stride=2), 'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 'layer18': nn.ReLU(inplace=True), 'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 'layer20': nn.ReLU(inplace=True), 'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 'layer22': nn.ReLU(inplace=True), 'layer23': nn.MaxPool2d(kernel_size=2, stride=2), 'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 'layer25': nn.ReLU(inplace=True), 'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 'layer27': nn.ReLU(inplace=True), 'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 'layer29': nn.ReLU(inplace=True), 'layer30': nn.MaxPool2d(kernel_size=2, stride=2)}))
        self.output_layer_idx = output_layer_idx
        self.min_val = min_val
        self.max_val = max_val
        self.mean = torch.from_numpy(np.array(_MEAN_STATS)).view(1, 3, 1, 1)
        self.mean = self.mean.type(torch.FloatTensor)
        self.pretrained_weight_path = pretrained_weight_path
        if os.path.isfile(self.pretrained_weight_path):
            self.vgg16.load_state_dict(torch.load(self.pretrained_weight_path, map_location='cpu'))
        else:
            warnings.warn('No pre-trained weights found for perceptual model!')

    def forward(self, x):
        x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
        x = x.flip(1)
        x = x - self.mean
        for idx, layer in enumerate(self.vgg16.children()):
            if idx == self.output_layer_idx:
                break
            x = layer(x)
        x = x.flatten(start_dim=1)
        return x


class Blur(torch.autograd.Function):
    """Defines blur operation with customized gradient computation."""

    @staticmethod
    def forward(ctx, x, kernel):
        ctx.save_for_backward(kernel)
        y = F.conv2d(input=x, weight=kernel, bias=None, stride=1, padding=1, groups=x.shape[1])
        return y

    @staticmethod
    def backward(ctx, dy):
        kernel, = ctx.saved_tensors
        dx = F.conv2d(input=dy, weight=kernel.flip((2, 3)), bias=None, stride=1, padding=1, groups=dy.shape[1])
        return dx, None, None


class BlurLayer(nn.Module):
    """Implements the blur layer."""

    def __init__(self, channels, kernel=(1, 2, 1), normalize=True):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        kernel = kernel[np.newaxis, np.newaxis]
        kernel = np.tile(kernel, [channels, 1, 1, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))

    def forward(self, x):
        return Blur.apply(x, self.kernel)


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f'The input tensor should be with shape [batch_size, channel, height, width], but `{x.shape}` is received!')
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        norm = torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.eps)
        return x / norm


class NoiseApplyingLayer(nn.Module):
    """Implements the noise applying layer."""

    def __init__(self, resolution, channels, noise_type='spatial'):
        super().__init__()
        self.noise_type = noise_type.lower()
        self.res = resolution
        self.channels = channels
        if self.noise_type == 'spatial':
            self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
            self.weight = nn.Parameter(torch.zeros(self.channels))
        elif self.noise_type == 'channel':
            self.register_buffer('noise', torch.randn(1, self.channels, 1, 1))
            self.weight = nn.Parameter(torch.zeros(self.res, self.res))
        else:
            raise NotImplementedError(f'Not implemented noise type: `{self.noise_type}`!')

    def forward(self, x, randomize_noise=False):
        if x.ndim != 4:
            raise ValueError(f'The input tensor should be with shape [batch_size, channel, height, width], but `{x.shape}` is received!')
        if randomize_noise:
            if self.noise_type == 'spatial':
                noise = torch.randn(x.shape[0], 1, self.res, self.res)
            elif self.noise_type == 'channel':
                noise = torch.randn(x.shape[0], self.channels, 1, 1)
        else:
            noise = self.noise
        if self.noise_type == 'spatial':
            x = x + noise * self.weight.view(1, self.channels, 1, 1)
        elif self.noise_type == 'channel':
            x = x + noise * self.weight.view(1, 1, self.res, self.res)
        return x


_STYLEMOD_WSCALE_GAIN = 1.0


class StyleModLayer(nn.Module):
    """Implements the style modulation layer."""

    def __init__(self, w_space_dim, out_channels, use_wscale=True):
        super().__init__()
        self.w_space_dim = w_space_dim
        self.out_channels = out_channels
        weight_shape = self.out_channels * 2, self.w_space_dim
        wscale = _STYLEMOD_WSCALE_GAIN / np.sqrt(self.w_space_dim)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape))
            self.wscale = wscale
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale)
            self.wscale = 1.0
        self.bias = nn.Parameter(torch.zeros(self.out_channels * 2))
        self.space_of_latent = 'w'

    def forward_style(self, w):
        """Gets style code from the given input.

        More specifically, if the input is from W-Space, it will be projected by
        an affine transformation. If it is from the Style Space (Y-Space), no
        operation is required.

        NOTE: For codes from Y-Space, we use slicing to make sure the dimension
        is correct, in case that the code is padded before fed into this layer.
        """
        if self.space_of_latent == 'w':
            if w.ndim != 2 or w.shape[1] != self.w_space_dim:
                raise ValueError(f'The input tensor should be with shape [batch_size, w_space_dim], where `w_space_dim` equals to {self.w_space_dim}!\nBut `{w.shape}` is received!')
            style = F.linear(w, weight=self.weight * self.wscale, bias=self.bias)
        elif self.space_of_latent == 'y':
            if w.ndim != 2 or w.shape[1] < 2 * self.out_channels:
                raise ValueError(f'The input tensor should be with shape [batch_size, y_space_dim], where `y_space_dim` equals to {2 * self.out_channels}!\nBut `{w.shape}` is received!')
            style = w[:, :2 * self.out_channels]
        return style

    def forward(self, x, w):
        style = self.forward_style(w)
        style_split = style.view(-1, 2, self.out_channels, 1, 1)
        x = x * (style_split[:, 0] + 1) + style_split[:, 1]
        return x, style


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    Basically, this layer can be used to upsample feature maps with nearest
    neighbor interpolation.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


_WSCALE_GAIN = np.sqrt(2.0)


class ConvBlock(nn.Module):
    """Implements the normal convolutional block.

    Basically, this block executes upsampling layer (if needed), convolutional
    layer, blurring layer, noise applying layer, activation layer, instance
    normalization layer, and style modulation layer in sequence.
    """

    def __init__(self, in_channels, out_channels, resolution, w_space_dim, position=None, kernel_size=3, stride=1, padding=1, add_bias=True, upsample=False, fused_scale=False, use_wscale=True, wscale_gain=_WSCALE_GAIN, lr_mul=1.0, activation_type='lrelu', noise_type='spatial'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_space_dim: Dimension of W space for style modulation.
            position: Position of the layer. `const_init`, `last` would lead to
                different behavior. (default: None)
            kernel_size: Size of the convolutional kernels. (default: 3)
            stride: Stride parameter for convolution operation. (default: 1)
            padding: Padding parameter for convolution operation. (default: 1)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            upsample: Whether to upsample the input tensor before convolution.
                (default: False)
            fused_scale: Whether to fused `upsample` and `conv2d` together,
                resulting in `conv2d_transpose`. (default: False)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `spatial` and `channel`.
                (default: `spatial`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        self.position = position
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: `{activation_type}`!')
        if self.position != 'last':
            self.apply_noise = NoiseApplyingLayer(resolution, out_channels, noise_type=noise_type)
            self.normalize = InstanceNormLayer()
            self.style = StyleModLayer(w_space_dim, out_channels, use_wscale)
        if self.position == 'const_init':
            self.const = nn.Parameter(torch.ones(1, in_channels, resolution, resolution))
            return
        self.blur = BlurLayer(out_channels) if upsample else nn.Identity()
        if upsample and not fused_scale:
            self.upsample = UpsamplingLayer()
        else:
            self.upsample = nn.Identity()
        if upsample and fused_scale:
            self.use_conv2d_transpose = True
            self.stride = 2
            self.padding = 1
        else:
            self.use_conv2d_transpose = False
            self.stride = stride
            self.padding = padding
        weight_shape = out_channels, in_channels, kernel_size, kernel_size
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

    def forward(self, x, w, randomize_noise=False):
        if self.position != 'const_init':
            x = self.upsample(x)
            weight = self.weight * self.wscale
            if self.use_conv2d_transpose:
                weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0)
                weight = weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]
                weight = weight.permute(1, 0, 2, 3)
                x = F.conv_transpose2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding)
            else:
                x = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding)
            x = self.blur(x)
        else:
            x = self.const.repeat(w.shape[0], 1, 1, 1)
        bias = self.bias * self.bscale if self.bias is not None else None
        if self.position == 'last':
            if bias is not None:
                x = x + bias.view(1, -1, 1, 1)
            return x
        x = self.apply_noise(x, randomize_noise)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x)
        x = self.normalize(x)
        x, style = self.style(x, w)
        return x, style


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer and activation layer.
    """

    def __init__(self, in_channels, out_channels, add_bias=True, use_wscale=True, wscale_gain=_WSCALE_GAIN, lr_mul=1.0, activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
                (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        weight_shape = out_channels, in_channels
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: `{activation_type}`!')

    def forward(self, x):
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        bias = self.bias * self.bscale if self.bias is not None else None
        x = F.linear(x, weight=self.weight * self.wscale, bias=bias)
        x = self.activate(x)
        return x


class DownsamplingLayer(nn.Module):
    """Implements the downsampling layer.

    Basically, this layer can be used to downsample feature maps with average
    pooling.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor, padding=0)


_INIT_RES = 4


class PGGANDiscriminator(nn.Module):
    """Defines the discriminator network in PGGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) image_channels: Number of channels of the input image. (default: 3)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4) fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: False)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) minibatch_std_group_size: Group size for the minibatch standard
        deviation layer. 0 means disable. (default: 16)
    (7) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (8) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, image_channels=3, label_size=0, fused_scale=False, use_wscale=True, minibatch_std_group_size=16, fmaps_base=16 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.image_channels = image_channels
        self.label_size = label_size
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.minibatch_std_group_size = minibatch_std_group_size
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            block_idx = self.final_res_log2 - res_log2
            self.add_module(f'input{block_idx}', ConvBlock(in_channels=self.image_channels, out_channels=self.get_nf(res), kernel_size=1, padding=0, use_wscale=self.use_wscale))
            self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = f'FromRGB_lod{block_idx}/weight'
            self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = f'FromRGB_lod{block_idx}/bias'
            if res != self.init_res:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale))
                tf_layer0_name = 'Conv0'
                self.add_module(f'layer{2 * block_idx + 1}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res // 2), downsample=True, fused_scale=self.fused_scale, use_wscale=self.use_wscale))
                tf_layer1_name = 'Conv1_down' if self.fused_scale else 'Conv1'
            else:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale, minibatch_std_group_size=self.minibatch_std_group_size))
                tf_layer0_name = 'Conv'
                self.add_module(f'layer{2 * block_idx + 1}', DenseBlock(in_channels=self.get_nf(res) * res * res, out_channels=self.get_nf(res // 2), use_wscale=self.use_wscale))
                tf_layer1_name = 'Dense0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = f'{res}x{res}/{tf_layer0_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = f'{res}x{res}/{tf_layer0_name}/bias'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = f'{res}x{res}/{tf_layer1_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = f'{res}x{res}/{tf_layer1_name}/bias'
        self.add_module(f'layer{2 * block_idx + 2}', DenseBlock(in_channels=self.get_nf(res // 2), out_channels=1 + self.label_size, use_wscale=self.use_wscale, wscale_gain=1.0, activation_type='linear'))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.weight'] = f'{res}x{res}/Dense1/weight'
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = f'{res}x{res}/Dense1/bias'
        self.downsample = DownsamplingLayer()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, lod=None, **_unused_kwargs):
        expected_shape = self.image_channels, self.resolution, self.resolution
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape [batch_size, channel, height, width], where `channel` equals to {self.image_channels}, `height`, `width` equal to {self.resolution}!\nBut `{image.shape}` is received!')
        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is {self.final_res_log2 - self.init_res_log2}, but `{lod}` is received!')
        lod = self.lod.cpu().tolist()
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = current_lod = self.final_res_log2 - res_log2
            if current_lod <= lod < current_lod + 1:
                x = self.__getattr__(f'input{block_idx}')(image)
            elif current_lod - 1 < lod < current_lod:
                alpha = lod - np.floor(lod)
                x = self.__getattr__(f'input{block_idx}')(image) * alpha + x * (1 - alpha)
            if lod < current_lod + 1:
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if lod > current_lod:
                image = self.downsample(image)
        x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
        return x


class MiniBatchSTDLayer(nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, group_size=4, new_channels=1, epsilon=1e-08):
        super().__init__()
        self.group_size = group_size
        self.new_channels = new_channels
        self.epsilon = epsilon

    def forward(self, x):
        if self.group_size <= 1:
            return x
        ng = min(self.group_size, x.shape[0])
        nc = self.new_channels
        temp_c = x.shape[1] // nc
        y = x.view(ng, -1, nc, temp_c, x.shape[2], x.shape[3])
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + self.epsilon)
        y = torch.mean(y, dim=[2, 3, 4], keepdim=True)
        y = torch.mean(y, dim=2)
        y = y.repeat(ng, 1, x.shape[2], x.shape[3])
        return torch.cat([x, y], dim=1)


class PGGANGenerator(nn.Module):
    """Defines the generator network in PGGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the network:

    (1) resolution: The resolution of the output image.
    (2) z_space_dim: The dimension of the latent space, Z. (default: 512)
    (3) image_channels: Number of channels of the output image. (default: 3)
    (4) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (5) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (6) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: False)
    (7) use_wscale: Whether to use weight scaling. (default: True)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, z_space_dim=512, image_channels=3, final_tanh=False, label_size=0, fused_scale=False, use_wscale=True, fmaps_base=16 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.z_space_dim = z_space_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.label_size = label_size
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2
            if res == self.init_res:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.z_space_dim + self.label_size, out_channels=self.get_nf(res), kernel_size=self.init_res, padding=self.init_res - 1, use_wscale=self.use_wscale))
                tf_layer_name = 'Dense'
            else:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res // 2), out_channels=self.get_nf(res), upsample=True, fused_scale=self.fused_scale, use_wscale=self.use_wscale))
                tf_layer_name = 'Conv0_up' if self.fused_scale else 'Conv0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = f'{res}x{res}/{tf_layer_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = f'{res}x{res}/{tf_layer_name}/bias'
            self.add_module(f'layer{2 * block_idx + 1}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = f'{res}x{res}/{tf_layer_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = f'{res}x{res}/{tf_layer_name}/bias'
            self.add_module(f'output{block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.image_channels, kernel_size=1, padding=0, use_wscale=self.use_wscale, wscale_gain=1.0, activation_type='linear'))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = f'ToRGB_lod{self.final_res_log2 - res_log2}/weight'
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = f'ToRGB_lod{self.final_res_log2 - res_log2}/bias'
        self.upsample = UpsamplingLayer()
        self.final_activate = nn.Tanh() if self.final_tanh else nn.Identity()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, z, label=None, lod=None, **_unused_kwargs):
        if z.ndim != 2 or z.shape[1] != self.z_space_dim:
            raise ValueError(f'Input latent code should be with shape [batch_size, latent_dim], where `latent_dim` equals to {self.z_space_dim}!\nBut `{z.shape}` is received!')
        z = self.layer0.pixel_norm(z)
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label (with size {self.label_size}) as input, but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
                raise ValueError(f'Input label should be with shape [batch_size, label_size], where `batch_size` equals to that of latent codes ({z.shape[0]}) and `label_size` equals to {self.label_size}!\nBut `{label.shape}` is received!')
            z = torch.cat((z, label), dim=1)
        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is {self.final_res_log2 - self.init_res_log2}, but `{lod}` is received!')
        x = z.view(z.shape[0], self.z_space_dim + self.label_size, 1, 1)
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            if lod < current_lod + 1:
                block_idx = res_log2 - self.init_res_log2
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if current_lod - 1 < lod <= current_lod:
                image = self.__getattr__(f'output{block_idx}')(x)
            elif current_lod < lod < current_lod + 1:
                alpha = np.ceil(lod) - lod
                image = self.__getattr__(f'output{block_idx}')(x) * alpha + self.upsample(image) * (1 - alpha)
            elif lod >= current_lod + 1:
                image = self.upsample(image)
        image = self.final_activate(image)
        results = {'z': z, 'label': label, 'image': image}
        return results


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / norm


_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']


class StyleGAN2Discriminator(nn.Module):
    """Defines the discriminator network in StyleGAN2.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) image_channels: Number of channels of the input image. (default: 3)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `resnet`)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) minibatch_std_group_size: Group size for the minibatch standard
        deviation layer. 0 means disable. (default: 4)
    (7) minibatch_std_channels: Number of new channels after the minibatch
        standard deviation layer. (default: 1)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 32 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, image_channels=3, label_size=0, architecture='resnet', use_wscale=True, minibatch_std_group_size=4, minibatch_std_channels=1, fmaps_base=32 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if architecture not in _ARCHITECTURES_ALLOWED:
            raise ValueError(f'Invalid architecture: `{architecture}`!\nArchitectures allowed: {_ARCHITECTURES_ALLOWED}.')
        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.image_channels = image_channels
        self.label_size = label_size
        self.architecture = architecture
        self.use_wscale = use_wscale
        self.minibatch_std_group_size = minibatch_std_group_size
        self.minibatch_std_channels = minibatch_std_channels
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.pth_to_tf_var_mapping = {}
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            block_idx = self.final_res_log2 - res_log2
            if res_log2 == self.final_res_log2 or self.architecture == 'skip':
                self.add_module(f'input{block_idx}', ConvBlock(in_channels=self.image_channels, out_channels=self.get_nf(res), kernel_size=1, use_wscale=self.use_wscale))
                self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = f'{res}x{res}/FromRGB/weight'
                self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = f'{res}x{res}/FromRGB/bias'
            if res != self.init_res:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale))
                tf_layer0_name = 'Conv0'
                self.add_module(f'layer{2 * block_idx + 1}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res // 2), scale_factor=2, use_wscale=self.use_wscale))
                tf_layer1_name = 'Conv1_down'
                if self.architecture == 'resnet':
                    layer_name = f'skip_layer{block_idx}'
                    self.add_module(layer_name, ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res // 2), kernel_size=1, add_bias=False, scale_factor=2, use_wscale=self.use_wscale, activation_type='linear'))
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = f'{res}x{res}/Skip/weight'
            else:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale, minibatch_std_group_size=minibatch_std_group_size, minibatch_std_channels=minibatch_std_channels))
                tf_layer0_name = 'Conv'
                self.add_module(f'layer{2 * block_idx + 1}', DenseBlock(in_channels=self.get_nf(res) * res * res, out_channels=self.get_nf(res // 2), use_wscale=self.use_wscale))
                tf_layer1_name = 'Dense0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = f'{res}x{res}/{tf_layer0_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = f'{res}x{res}/{tf_layer0_name}/bias'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = f'{res}x{res}/{tf_layer1_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = f'{res}x{res}/{tf_layer1_name}/bias'
            self.add_module(f'layer{2 * block_idx + 2}', DenseBlock(in_channels=self.get_nf(res // 2), out_channels=max(self.label_size, 1), use_wscale=self.use_wscale, activation_type='linear'))
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.weight'] = f'Output/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = f'Output/bias'
        if self.architecture == 'skip':
            self.downsample = DownsamplingLayer()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, label=None, **_unused_kwargs):
        expected_shape = self.image_channels, self.resolution, self.resolution
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape [batch_size, channel, height, width], where `channel` equals to {self.image_channels}, `height`, `width` equal to {self.resolution}!\nBut `{image.shape}` is received!')
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label (with size {self.label_size}) as inputs, but no label is received!')
            batch_size = image.shape[0]
            if label.ndim != 2 or label.shape != (batch_size, self.label_size):
                raise ValueError(f'Input label should be with shape [batch_size, label_size], where `batch_size` equals to that of images ({image.shape[0]}) and `label_size` equals to {self.label_size}!\nBut `{label.shape}` is received!')
        x = self.input0(image)
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = self.final_res_log2 - res_log2
            if self.architecture == 'skip' and block_idx > 0:
                image = self.downsample(image)
                x = x + self.__getattr__(f'input{block_idx}')(image)
            if self.architecture == 'resnet' and res_log2 != self.init_res_log2:
                residual = self.__getattr__(f'skip_layer{block_idx}')(x)
            x = self.__getattr__(f'layer{2 * block_idx}')(x)
            x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if self.architecture == 'resnet' and res_log2 != self.init_res_log2:
                x = (x + residual) / np.sqrt(2.0)
        x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
        if self.label_size:
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x


class MappingModule(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(self, input_space_dim=512, hidden_space_dim=512, final_space_dim=512, label_size=0, num_layers=8, normalize_input=True, use_wscale=True, lr_mul=0.01):
        super().__init__()
        self.input_space_dim = input_space_dim
        self.hidden_space_dim = hidden_space_dim
        self.final_space_dim = final_space_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul
        self.norm = PixelNormLayer() if self.normalize_input else nn.Identity()
        self.pth_to_tf_var_mapping = {}
        for i in range(num_layers):
            dim_mul = 2 if label_size else 1
            in_channels = input_space_dim * dim_mul if i == 0 else hidden_space_dim
            out_channels = final_space_dim if i == num_layers - 1 else hidden_space_dim
            self.add_module(f'dense{i}', DenseBlock(in_channels=in_channels, out_channels=out_channels, use_wscale=self.use_wscale, lr_mul=self.lr_mul))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'
        if label_size:
            self.label_weight = nn.Parameter(torch.randn(label_size, input_space_dim))
            self.pth_to_tf_var_mapping[f'label_weight'] = f'LabelConcat/weight'

    def forward(self, z, label=None):
        if z.ndim != 2 or z.shape[1] != self.input_space_dim:
            raise ValueError(f'Input latent code should be with shape [batch_size, input_dim], where `input_dim` equals to {self.input_space_dim}!\nBut `{z.shape}` is received!')
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label (with size {self.label_size}) as input, but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
                raise ValueError(f'Input label should be with shape [batch_size, label_size], where `batch_size` equals to that of latent codes ({z.shape[0]}) and `label_size` equals to {self.label_size}!\nBut `{label.shape}` is received!')
            embedding = torch.matmul(label, self.label_weight)
            z = torch.cat((z, embedding), dim=1)
        z = self.norm(z)
        w = z
        for i in range(self.num_layers):
            w = self.__getattr__(f'dense{i}')(w)
        results = {'z': z, 'label': label, 'w': w}
        if self.label_size:
            results['embedding'] = embedding
        return results


class ModulateConvBlock(nn.Module):
    """Implements the convolutional block with style modulation."""

    def __init__(self, in_channels, out_channels, resolution, w_space_dim, kernel_size=3, add_bias=True, scale_factor=1, filtering_kernel=(1, 3, 3, 1), fused_modulate=True, demodulate=True, use_wscale=True, wscale_gain=_WSCALE_GAIN, lr_mul=1.0, add_noise=True, noise_type='spatial', activation_type='lrelu', epsilon=1e-08):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_space_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels. (default: 3)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            scale_factor: Scale factor for upsampling. `1` means skip
                upsampling. (default: 1)
            filtering_kernel: Kernel used for filtering after upsampling.
                (default: (1, 3, 3, 1))
            fused_modulate: Whether to fuse `style_modulate` and `conv2d`
                together. (default: True)
            demodulate: Whether to perform style demodulation. (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            add_noise: Whether to add noise onto the output tensor. (default:
                True)
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `spatial` and `channel`.
                (default: `spatial`)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            epsilon: Small number to avoid `divide by zero`. (default: 1e-8)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.res = resolution
        self.w_space_dim = w_space_dim
        self.ksize = kernel_size
        self.eps = epsilon
        self.space_of_latent = 'w'
        if scale_factor > 1:
            self.use_conv2d_transpose = True
            extra_padding = scale_factor - kernel_size
            self.filter = UpsamplingLayer(scale_factor=1, kernel=filtering_kernel, extra_padding=extra_padding, kernel_gain=scale_factor)
            self.stride = scale_factor
            self.padding = 0
        else:
            self.use_conv2d_transpose = False
            assert kernel_size % 2 == 1
            self.stride = 1
            self.padding = kernel_size // 2
        weight_shape = out_channels, in_channels, kernel_size, kernel_size
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul
        self.style = DenseBlock(in_channels=w_space_dim, out_channels=in_channels, additional_bias=1.0, use_wscale=use_wscale, activation_type='linear')
        self.fused_modulate = fused_modulate
        self.demodulate = demodulate
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.bscale = lr_mul
        if activation_type == 'linear':
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f'Not implemented activation function: `{activation_type}`!')
        self.add_noise = add_noise
        if self.add_noise:
            self.noise_type = noise_type.lower()
            if self.noise_type == 'spatial':
                self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
            elif self.noise_type == 'channel':
                self.register_buffer('noise', torch.randn(1, self.channels, 1, 1))
            else:
                raise NotImplementedError(f'Not implemented noise type: `{self.noise_type}`!')
            self.noise_strength = nn.Parameter(torch.zeros(()))

    def forward_style(self, w):
        """Gets style code from the given input.

        More specifically, if the input is from W-Space, it will be projected by
        an affine transformation. If it is from the Style Space (Y-Space), no
        operation is required.

        NOTE: For codes from Y-Space, we use slicing to make sure the dimension
        is correct, in case that the code is padded before fed into this layer.
        """
        if self.space_of_latent == 'w':
            if w.ndim != 2 or w.shape[1] != self.w_space_dim:
                raise ValueError(f'The input tensor should be with shape [batch_size, w_space_dim], where `w_space_dim` equals to {self.w_space_dim}!\nBut `{w.shape}` is received!')
            style = self.style(w)
        elif self.space_of_latent == 'y':
            if w.ndim != 2 or w.shape[1] < self.in_c:
                raise ValueError(f'The input tensor should be with shape [batch_size, y_space_dim], where `y_space_dim` equals to {self.in_c}!\nBut `{w.shape}` is received!')
            style = w[:, :self.in_c]
        return style

    def forward(self, x, w, randomize_noise=False):
        batch = x.shape[0]
        weight = self.weight * self.wscale
        weight = weight.permute(2, 3, 1, 0)
        style = self.forward_style(w)
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * style.view(batch, 1, 1, self.in_c, 1)
        if self.demodulate:
            _weight_norm = torch.sqrt(torch.sum(_weight ** 2, dim=[1, 2, 3]) + self.eps)
            _weight = _weight / _weight_norm.view(batch, 1, 1, 1, self.out_c)
        if self.fused_modulate:
            x = x.view(1, batch * self.in_c, x.shape[2], x.shape[3])
            weight = _weight.permute(1, 2, 3, 0, 4).reshape(self.ksize, self.ksize, self.in_c, batch * self.out_c)
        else:
            x = x * style.view(batch, self.in_c, 1, 1)
        if self.use_conv2d_transpose:
            weight = weight.flip(0, 1)
            if self.fused_modulate:
                weight = weight.view(self.ksize, self.ksize, self.in_c, batch, self.out_c)
                weight = weight.permute(0, 1, 4, 3, 2)
                weight = weight.reshape(self.ksize, self.ksize, self.out_c, batch * self.in_c)
                weight = weight.permute(3, 2, 0, 1)
            else:
                weight = weight.permute(2, 3, 0, 1)
            x = F.conv_transpose2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=batch if self.fused_modulate else 1)
            x = self.filter(x)
        else:
            weight = weight.permute(3, 2, 0, 1)
            x = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=batch if self.fused_modulate else 1)
        if self.fused_modulate:
            x = x.view(batch, self.out_c, self.res, self.res)
        elif self.demodulate:
            x = x / _weight_norm.view(batch, self.out_c, 1, 1)
        if self.add_noise:
            if randomize_noise:
                if self.noise_type == 'spatial':
                    noise = torch.randn(x.shape[0], 1, self.res, self.res)
                elif self.noise_type == 'channel':
                    noise = torch.randn(x.shape[0], self.channels, 1, 1)
            else:
                noise = self.noise
            x = x + noise * self.noise_strength.view(1, 1, 1, 1)
        bias = self.bias * self.bscale if self.bias is not None else None
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale
        return x, style


_AUTO_FUSED_SCALE_MIN_RES = 128


class SynthesisModule(nn.Module):
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(self, resolution=1024, init_resolution=4, w_space_dim=512, image_channels=3, final_tanh=False, const_input=True, fused_scale='auto', use_wscale=True, noise_type='spatial', fmaps_base=16 << 10, fmaps_max=512):
        super().__init__()
        self.init_res = init_resolution
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2
            layer_name = f'layer{2 * block_idx}'
            if res == self.init_res:
                if self.const_input:
                    self.add_module(layer_name, ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), resolution=self.init_res, w_space_dim=self.w_space_dim, position='const_init', use_wscale=self.use_wscale, noise_type=self.noise_type))
                    tf_layer_name = 'Const'
                    self.pth_to_tf_var_mapping[f'{layer_name}.const'] = f'{res}x{res}/{tf_layer_name}/const'
                else:
                    self.add_module(layer_name, ConvBlock(in_channels=self.w_space_dim, out_channels=self.get_nf(res), resolution=self.init_res, w_space_dim=self.w_space_dim, kernel_size=self.init_res, padding=self.init_res - 1, use_wscale=self.use_wscale, noise_type=self.noise_type))
                    tf_layer_name = 'Dense'
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = f'{res}x{res}/{tf_layer_name}/weight'
            else:
                if self.fused_scale == 'auto':
                    fused_scale = res >= _AUTO_FUSED_SCALE_MIN_RES
                else:
                    fused_scale = self.fused_scale
                self.add_module(layer_name, ConvBlock(in_channels=self.get_nf(res // 2), out_channels=self.get_nf(res), resolution=res, w_space_dim=self.w_space_dim, upsample=True, fused_scale=fused_scale, use_wscale=self.use_wscale, noise_type=self.noise_type))
                tf_layer_name = 'Conv0_up'
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = f'{res}x{res}/{tf_layer_name}/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = f'{res}x{res}/{tf_layer_name}/bias'
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = f'{res}x{res}/{tf_layer_name}/StyleMod/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = f'{res}x{res}/{tf_layer_name}/StyleMod/bias'
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.weight'] = f'{res}x{res}/{tf_layer_name}/Noise/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.noise'] = f'noise{2 * block_idx}'
            layer_name = f'layer{2 * block_idx + 1}'
            self.add_module(layer_name, ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), resolution=res, w_space_dim=self.w_space_dim, use_wscale=self.use_wscale, noise_type=self.noise_type))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = f'{res}x{res}/{tf_layer_name}/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = f'{res}x{res}/{tf_layer_name}/bias'
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = f'{res}x{res}/{tf_layer_name}/StyleMod/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = f'{res}x{res}/{tf_layer_name}/StyleMod/bias'
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.weight'] = f'{res}x{res}/{tf_layer_name}/Noise/weight'
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.noise'] = f'noise{2 * block_idx + 1}'
            self.add_module(f'output{block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.image_channels, resolution=res, w_space_dim=self.w_space_dim, position='last', kernel_size=1, padding=0, use_wscale=self.use_wscale, wscale_gain=1.0, activation_type='linear'))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = f'ToRGB_lod{self.final_res_log2 - res_log2}/weight'
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = f'ToRGB_lod{self.final_res_log2 - res_log2}/bias'
        self.upsample = UpsamplingLayer()
        self.final_activate = nn.Tanh() if final_tanh else nn.Identity()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, wp, lod=None, randomize_noise=False):
        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is {self.final_res_log2 - self.init_res_log2}, but `{lod}` is received!')
        results = {'wp': wp}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            if lod < current_lod + 1:
                block_idx = res_log2 - self.init_res_log2
                if block_idx == 0:
                    if self.const_input:
                        x, style = self.layer0(None, wp[:, 0], randomize_noise)
                    else:
                        x = wp[:, 0].view(-1, self.w_space_dim, 1, 1)
                        x, style = self.layer0(x, wp[:, 0], randomize_noise)
                else:
                    x, style = self.__getattr__(f'layer{2 * block_idx}')(x, wp[:, 2 * block_idx], randomize_noise)
                results[f'style{2 * block_idx:02d}'] = style
                x, style = self.__getattr__(f'layer{2 * block_idx + 1}')(x, wp[:, 2 * block_idx + 1], randomize_noise)
                results[f'style{2 * block_idx + 1:02d}'] = style
            if current_lod - 1 < lod <= current_lod:
                image = self.__getattr__(f'output{block_idx}')(x, None)
            elif current_lod < lod < current_lod + 1:
                alpha = np.ceil(lod) - lod
                image = self.__getattr__(f'output{block_idx}')(x, None) * alpha + self.upsample(image) * (1 - alpha)
            elif lod >= current_lod + 1:
                image = self.upsample(image)
        results['image'] = self.final_activate(image)
        return results


class TruncationModule(nn.Module):
    """Implements the truncation module.

    Truncation is executed as follows:

    For layers in range [0, truncation_layers), the truncated w-code is computed
    as

    w_new = w_avg + (w - w_avg) * truncation_psi

    To disable truncation, please set
    (1) truncation_psi = 1.0 (None) OR
    (2) truncation_layers = 0 (None)

    NOTE: The returned tensor is layer-wise style codes.
    """

    def __init__(self, w_space_dim, num_layers, repeat_w=True):
        super().__init__()
        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        self.repeat_w = repeat_w
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_space_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(num_layers * w_space_dim))
        self.pth_to_tf_var_mapping = {'w_avg': 'dlatent_avg'}

    def forward(self, w, trunc_psi=None, trunc_layers=None):
        if w.ndim == 2:
            if self.repeat_w and w.shape[1] == self.w_space_dim:
                w = w.view(-1, 1, self.w_space_dim)
                wp = w.repeat(1, self.num_layers, 1)
            else:
                assert w.shape[1] == self.w_space_dim * self.num_layers
                wp = w.view(-1, self.num_layers, self.w_space_dim)
        else:
            wp = w
        assert wp.ndim == 3
        assert wp.shape[1:] == (self.num_layers, self.w_space_dim)
        trunc_psi = 1.0 if trunc_psi is None else trunc_psi
        trunc_layers = 0 if trunc_layers is None else trunc_layers
        if trunc_psi < 1.0 and trunc_layers > 0:
            layer_idx = np.arange(self.num_layers).reshape(1, -1, 1)
            coefs = np.ones_like(layer_idx, dtype=np.float32)
            coefs[layer_idx < trunc_layers] *= trunc_psi
            coefs = torch.from_numpy(coefs)
            w_avg = self.w_avg.view(1, -1, self.w_space_dim)
            wp = w_avg + (wp - w_avg) * coefs
        return wp


def all_gather(tensor):
    """Gathers tensor from all devices and does averaging."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, async_op=False)
    return torch.mean(torch.stack(tensor_list, dim=0), dim=0)


class StyleGAN2Generator(nn.Module):
    """Defines the generator network in StyleGAN2.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_space_dim: Dimension of the outout latent space, W. (default: 512)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `resnet`)
    (6) fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
        (default: True)
    (7) demodulate: Whether to perform style demodulation. (default: True)
    (8) use_wscale: Whether to use weight scaling. (default: True)
    (9) noise_type: Type of noise added to the convolutional results at each
        layer. (default: `spatial`)
    (10) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 32 << 10)
    (11) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, z_space_dim=512, w_space_dim=512, label_size=0, mapping_layers=8, mapping_fmaps=512, mapping_lr_mul=0.01, repeat_w=True, image_channels=3, final_tanh=False, const_input=True, architecture='skip', fused_modulate=True, demodulate=True, use_wscale=True, noise_type='spatial', fmaps_base=32 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if architecture not in _ARCHITECTURES_ALLOWED:
            raise ValueError(f'Invalid architecture: `{architecture}`!\nArchitectures allowed: {_ARCHITECTURES_ALLOWED}.')
        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul
        self.repeat_w = repeat_w
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.architecture = architecture
        self.fused_modulate = fused_modulate
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2
        if self.repeat_w:
            self.mapping_space_dim = self.w_space_dim
        else:
            self.mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModule(input_space_dim=self.z_space_dim, hidden_space_dim=self.mapping_fmaps, final_space_dim=self.mapping_space_dim, label_size=self.label_size, num_layers=self.mapping_layers, use_wscale=self.use_wscale, lr_mul=self.mapping_lr_mul)
        self.truncation = TruncationModule(w_space_dim=self.w_space_dim, num_layers=self.num_layers, repeat_w=self.repeat_w)
        self.synthesis = SynthesisModule(resolution=self.resolution, init_resolution=self.init_res, w_space_dim=self.w_space_dim, image_channels=self.image_channels, final_tanh=self.final_tanh, const_input=self.const_input, architecture=self.architecture, fused_modulate=self.fused_modulate, demodulate=self.demodulate, use_wscale=self.use_wscale, noise_type=self.noise_type, fmaps_base=self.fmaps_base, fmaps_max=self.fmaps_max)
        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def set_space_of_latent(self, space_of_latent='w'):
        """Sets the space to which the latent code belong.

        This function is particually used for choosing how to inject the latent
        code into the convolutional layers. The original generator will take a
        W-Space code and apply it for style modulation after an affine
        transformation. But, sometimes, it may need to directly feed an already
        affine-transformed code into the convolutional layer, e.g., when
        training an encoder for GAN inversion. We term the transformed space as
        Style Space (or Y-Space). This function is designed to tell the
        convolutional layers how to use the input code.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. (default: 'w')
        """
        for module in self.modules():
            if isinstance(module, ModulateConvBlock):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self, z, label=None, w_moving_decay=0.995, style_mixing_prob=0.9, trunc_psi=None, trunc_layers=None, randomize_noise=False, **_unused_kwargs):
        mapping_results = self.mapping(z, label)
        w = mapping_results['w']
        if self.training and w_moving_decay < 1:
            batch_w_avg = all_gather(w).mean(dim=0)
            self.truncation.w_avg.copy_(self.truncation.w_avg * w_moving_decay + batch_w_avg * (1 - w_moving_decay))
        if self.training and style_mixing_prob > 0:
            new_z = torch.randn_like(z)
            new_w = self.mapping(new_z, label)['w']
            if np.random.uniform() < style_mixing_prob:
                mixing_cutoff = np.random.randint(1, self.num_layers)
                w = self.truncation(w)
                new_w = self.truncation(new_w)
                w[:, :mixing_cutoff] = new_w[:, :mixing_cutoff]
        wp = self.truncation(w, trunc_psi, trunc_layers)
        synthesis_results = self.synthesis(wp, randomize_noise)
        return {**mapping_results, **synthesis_results}


class InputBlock(nn.Module):
    """Implements the input block.

    Basically, this block starts from a const input, which is with shape
    `(channels, init_resolution, init_resolution)`.
    """

    def __init__(self, init_resolution, channels):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, init_resolution, init_resolution))

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        return x


_FUSED_SCALE_ALLOWED = [True, False, 'auto']


class StyleGANDiscriminator(nn.Module):
    """Defines the discriminator network in StyleGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) image_channels: Number of channels of the input image. (default: 3)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4) fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: `auto`)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) minibatch_std_group_size: Group size for the minibatch standard
        deviation layer. 0 means disable. (default: 4)
    (7) minibatch_std_channels: Number of new channels after the minibatch
        standard deviation layer. (default: 1)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, image_channels=3, label_size=0, fused_scale='auto', use_wscale=True, minibatch_std_group_size=4, minibatch_std_channels=1, fmaps_base=16 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if fused_scale not in _FUSED_SCALE_ALLOWED:
            raise ValueError(f'Invalid fused-scale option: `{fused_scale}`!\nOptions allowed: {_FUSED_SCALE_ALLOWED}.')
        self.init_res = _INIT_RES
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.image_channels = image_channels
        self.label_size = label_size
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.minibatch_std_group_size = minibatch_std_group_size
        self.minibatch_std_channels = minibatch_std_channels
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            block_idx = self.final_res_log2 - res_log2
            self.add_module(f'input{block_idx}', ConvBlock(in_channels=self.image_channels, out_channels=self.get_nf(res), kernel_size=1, padding=0, use_wscale=self.use_wscale))
            self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = f'FromRGB_lod{block_idx}/weight'
            self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = f'FromRGB_lod{block_idx}/bias'
            if res != self.init_res:
                if self.fused_scale == 'auto':
                    fused_scale = res >= _AUTO_FUSED_SCALE_MIN_RES
                else:
                    fused_scale = self.fused_scale
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale))
                tf_layer0_name = 'Conv0'
                self.add_module(f'layer{2 * block_idx + 1}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res // 2), downsample=True, fused_scale=fused_scale, use_wscale=self.use_wscale))
                tf_layer1_name = 'Conv1_down'
            else:
                self.add_module(f'layer{2 * block_idx}', ConvBlock(in_channels=self.get_nf(res), out_channels=self.get_nf(res), use_wscale=self.use_wscale, minibatch_std_group_size=minibatch_std_group_size, minibatch_std_channels=minibatch_std_channels))
                tf_layer0_name = 'Conv'
                self.add_module(f'layer{2 * block_idx + 1}', DenseBlock(in_channels=self.get_nf(res) * res * res, out_channels=self.get_nf(res // 2), use_wscale=self.use_wscale))
                tf_layer1_name = 'Dense0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = f'{res}x{res}/{tf_layer0_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = f'{res}x{res}/{tf_layer0_name}/bias'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = f'{res}x{res}/{tf_layer1_name}/weight'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = f'{res}x{res}/{tf_layer1_name}/bias'
        self.add_module(f'layer{2 * block_idx + 2}', DenseBlock(in_channels=self.get_nf(res // 2), out_channels=max(self.label_size, 1), use_wscale=self.use_wscale, wscale_gain=1.0, activation_type='linear'))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.weight'] = f'{res}x{res}/Dense1/weight'
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 2}.bias'] = f'{res}x{res}/Dense1/bias'
        self.downsample = DownsamplingLayer()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, label=None, lod=None, **_unused_kwargs):
        expected_shape = self.image_channels, self.resolution, self.resolution
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape [batch_size, channel, height, width], where `channel` equals to {self.image_channels}, `height`, `width` equal to {self.resolution}!\nBut `{image.shape}` is received!')
        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is {self.final_res_log2 - self.init_res_log2}, but `{lod}` is received!')
        if self.label_size:
            if label is None:
                raise ValueError(f'Model requires an additional label (with size {self.label_size}) as input, but no label is received!')
            batch_size = image.shape[0]
            if label.ndim != 2 or label.shape != (batch_size, self.label_size):
                raise ValueError(f'Input label should be with shape [batch_size, label_size], where `batch_size` equals to that of images ({image.shape[0]}) and `label_size` equals to {self.label_size}!\nBut `{label.shape}` is received!')
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = current_lod = self.final_res_log2 - res_log2
            if current_lod <= lod < current_lod + 1:
                x = self.__getattr__(f'input{block_idx}')(image)
            elif current_lod - 1 < lod < current_lod:
                alpha = lod - np.floor(lod)
                x = self.__getattr__(f'input{block_idx}')(image) * alpha + x * (1 - alpha)
            if lod < current_lod + 1:
                x = self.__getattr__(f'layer{2 * block_idx}')(x)
                x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
            if lod > current_lod:
                image = self.downsample(image)
        x = self.__getattr__(f'layer{2 * block_idx + 2}')(x)
        if self.label_size:
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x


class StyleGANGenerator(nn.Module):
    """Defines the generator network in StyleGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_space_dim: Dimension of the outout latent space, W. (default: 512)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: `auto`)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) noise_type: Type of noise added to the convolutional results at each
        layer. (default: `spatial`)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self, resolution, z_space_dim=512, w_space_dim=512, label_size=0, mapping_layers=8, mapping_fmaps=512, mapping_lr_mul=0.01, repeat_w=True, image_channels=3, final_tanh=False, const_input=True, fused_scale='auto', use_wscale=True, noise_type='spatial', fmaps_base=16 << 10, fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if fused_scale not in _FUSED_SCALE_ALLOWED:
            raise ValueError(f'Invalid fused-scale option: `{fused_scale}`!\nOptions allowed: {_FUSED_SCALE_ALLOWED}.')
        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul
        self.repeat_w = repeat_w
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2
        if self.repeat_w:
            self.mapping_space_dim = self.w_space_dim
        else:
            self.mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModule(input_space_dim=self.z_space_dim, hidden_space_dim=self.mapping_fmaps, final_space_dim=self.mapping_space_dim, label_size=self.label_size, num_layers=self.mapping_layers, use_wscale=self.use_wscale, lr_mul=self.mapping_lr_mul)
        self.truncation = TruncationModule(w_space_dim=self.w_space_dim, num_layers=self.num_layers, repeat_w=self.repeat_w)
        self.synthesis = SynthesisModule(resolution=self.resolution, init_resolution=self.init_res, w_space_dim=self.w_space_dim, image_channels=self.image_channels, final_tanh=self.final_tanh, const_input=self.const_input, fused_scale=self.fused_scale, use_wscale=self.use_wscale, noise_type=self.noise_type, fmaps_base=self.fmaps_base, fmaps_max=self.fmaps_max)
        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def set_space_of_latent(self, space_of_latent='w'):
        """Sets the space to which the latent code belong.

        This function is particually used for choosing how to inject the latent
        code into the convolutional layers. The original generator will take a
        W-Space code and apply it for style modulation after an affine
        transformation. But, sometimes, it may need to directly feed an already
        affine-transformed code into the convolutional layer, e.g., when
        training an encoder for GAN inversion. We term the transformed space as
        Style Space (or Y-Space). This function is designed to tell the
        convolutional layers how to use the input code.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. (default: 'w')
        """
        for module in self.modules():
            if isinstance(module, StyleModLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self, z, label=None, lod=None, w_moving_decay=0.995, style_mixing_prob=0.9, trunc_psi=None, trunc_layers=None, randomize_noise=False, **_unused_kwargs):
        mapping_results = self.mapping(z, label)
        w = mapping_results['w']
        if self.training and w_moving_decay < 1:
            batch_w_avg = all_gather(w).mean(dim=0)
            self.truncation.w_avg.copy_(self.truncation.w_avg * w_moving_decay + batch_w_avg * (1 - w_moving_decay))
        if self.training and style_mixing_prob > 0:
            new_z = torch.randn_like(z)
            new_w = self.mapping(new_z, label)['w']
            lod = self.synthesis.lod.cpu().tolist() if lod is None else lod
            current_layers = self.num_layers - int(lod) * 2
            if np.random.uniform() < style_mixing_prob:
                mixing_cutoff = np.random.randint(1, current_layers)
                w = self.truncation(w)
                new_w = self.truncation(new_w)
                w[:, mixing_cutoff:] = new_w[:, mixing_cutoff:]
        wp = self.truncation(w, trunc_psi, trunc_layers)
        synthesis_results = self.synthesis(wp, lod, randomize_noise)
        return {**mapping_results, **synthesis_results}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlurLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CodeHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DenseBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DownsamplingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageNormalization,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ImageResizing,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Inception3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionAux,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputBlock,
     lambda: ([], {'init_resolution': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InstanceNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LatentCodeNormalization,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MiniBatchSTDLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoiseApplyingLayer,
     lambda: ([], {'resolution': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyleModLayer,
     lambda: ([], {'w_space_dim': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (TruncationModule,
     lambda: ([], {'w_space_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (UpsamplingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_genforce_genforce(_paritybench_base):
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

