import sys
_module = sys.modules[__name__]
del sys
brax = _module
_impl = _module
arch = _module
conv = _module
diffeq_layers = _module
layers = _module
resnet = _module
test_sdeint = _module
utils = _module
datasets = _module
registry = _module
sdeint = _module
stl = _module
sdebnn_toy1d = _module
latent_sde = _module
sdebnn_classification = _module
sdebnn_toy1d = _module
demo = _module
demo_fokker = _module
demo_svi = _module
jaxsde = _module
brownian = _module
sde_jvp = _module
sde_utils = _module
sde_vjp = _module
sdeint_wrapper = _module
svi = _module
tests = _module
test_brownian = _module
test_jvp = _module
test_utils = _module
test_vjp = _module
torchbnn = _module
basic = _module
container = _module
diffeq_layers = _module
models = _module
resnet = _module
utils = _module
wrappers = _module
test_diffeq_layers = _module
test_utils = _module

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


import numpy as onp


import numpy.random as npr


import tensorflow as tf


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import CIFAR10


from torchvision.datasets import MNIST


import torch


import copy


import logging


import math


import numpy as np


from torch import nn


from torch import optim


import random


from collections import namedtuple


from typing import Optional


from typing import Union


import matplotlib.pyplot as plt


from torch import distributions


from torch.utils.tensorboard import SummaryWriter


import collections


import time


import torch.nn.functional as F


import matplotlib._color_data as mcd


import torch.nn as nn


import abc


from typing import Any


from typing import List


from torch.nn.common_types import _size_2_t


from torch.nn.modules.utils import _pair


from typing import Callable


from typing import Sequence


from typing import Tuple


import matplotlib.gridspec as gridspec


import torchvision as tv


from scipy import stats


from torch.utils import data


from inspect import signature


class SDE(nn.Module):

    def __init__(self, mu=0.0, sigma=1.0):
        super(SDE, self).__init__()
        self.mu = mu
        self.sigma = sigma
        net = nn.Sequential(nn.Linear(3, 400), nn.Tanh(), nn.Linear(400, 1))
        self.net = net

    def f(self, t, y):
        t = float(t) * torch.ones_like(y)
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        return self.net(inp)

    def g(self, t, y):
        return self.sigma * torch.ones_like(y)

    def h(self, t, y):
        return self.mu * y

    def f_detach(self, t, y):
        t = float(t) * torch.ones_like(y)
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        net = copy.deepcopy(self.net)
        return net(inp)


def em_solver(f, g, f_p, z0, t):
    z_t = [z0]
    kldiv = torch.zeros(1)
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        dW = torch.randn_like(z0) * torch.sqrt(dt)
        f_t = f(t0, z_t[-1])
        g_t = g(t0, z_t[-1])
        z_t.append(z_t[-1] + f_t * dt + g_t * dW)
        u = (f_t - f_p(t0, z_t[-1])) / g_t
        kldiv = kldiv + dt * torch.abs(u * u) * 0.5
    return z_t, kldiv


class VariationalSDE(nn.Module):

    def __init__(self, f_func, ou_drift_coef=0.1, diff_coef=1.0 * math.sqrt(2.0)):
        super().__init__()
        self.f_func = f_func
        self.g_func = lambda t, x: diff_coef * torch.ones_like(x)
        self.f_prior_func = lambda t, x: -ou_drift_coef * x
        self.start_time = 0.0
        self.end_time = 3.0

    def forward(self, nsamples):
        z0 = torch.randn(nsamples, 1) * 3
        z_t, kldiv = em_solver(self.f_func, self.g_func, self.f_prior_func, z0, torch.linspace(self.start_time, self.end_time, 100))
        return torch.stack(z_t, dim=0), kldiv


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):

    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out
        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class DiffEqModule(abc.ABC, nn.Module):

    def make_initial_params(self):
        return [p.detach().clone() for p in self.parameters()]

    def _forward_unimplemented(self, *input: Any) ->None:
        pass


class ConcatLinear(nn.Linear, DiffEqModule):

    def __init__(self, in_features: int, out_features: int):
        super(ConcatLinear, self).__init__(in_features=in_features + 1, out_features=out_features)

    def forward(self, t, y, params: Optional[List]=None):
        w, b = (self.weight, self.bias) if params is None else params
        ty = utils.channel_cat(t, y)
        return F.linear(ty, w, b)


class ConcatLinear_v2(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class SquashLinear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(-1, 1)))


class ConcatSquashLinear(nn.Linear, DiffEqModule):

    def __init__(self, in_features: int, out_features: int):
        super(ConcatSquashLinear, self).__init__(in_features=in_features + 1, out_features=out_features)
        self.wt = nn.Parameter(torch.randn(1, self.out_features))
        self.by = nn.Parameter(torch.randn(1, self.out_features))
        self.b = nn.Parameter(torch.randn(1, self.out_features))

    def forward(self, t, y, params: Optional[List]=None):
        wy, by, wt, bt, b = (self.weight, self.bias, self.wt, self.bt, self.b) if params is None else params
        ty = utils.channel_cat(t, y)
        net = F.linear(ty, wy, by)
        scale = torch.sigmoid(t * wt + bt)
        shift = b * t
        return scale * net + shift

    def make_initial_params(self):
        return [self.weight.clone(), self.bias.clone(), torch.randn(1, self.out_features), torch.randn(1, self.out_features), torch.randn(1, self.out_features)]


class HyperConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, 'dim_in and dim_out must both be divisible by groups.'
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose
        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)


class IgnoreConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(IgnoreConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, t, x):
        return self._layer(x)


class SquashConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(nn.Conv2d, DiffEqModule):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t=0, dilation: _size_2_t=1, groups: int=1, bias: bool=True, padding_mode: str='zeros'):
        super(ConcatConv2d, self).__init__(in_channels=in_channels + 1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.unpack_params = self.unpack_wb if bias else self.unpack_w

    def forward(self, t, y, params: Optional[List]=None):
        weight, bias = self.unpack_params(params)
        ty = utils.channel_cat(t, y)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(ty, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(ty, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def unpack_wb(self, params: Optional[List]=None):
        if params is None:
            return self.weight, self.bias
        return params

    def unpack_w(self, params: Optional[List]=None):
        if params is None:
            return self.weight, self.bias
        return params[0], None


class ConcatConv2d_v2(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(1, -1, 1, 1) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups)
        self.layer_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups)
        self.layer_g = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):

    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False, **unused_kwargs):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._layer1 = module(dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class TimeDependentSwish(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(nn.Linear(1, min(64, dim * 4)), nn.Softplus(), nn.Linear(min(64, dim * 4), dim), nn.Softplus())

    def forward(self, t, x):
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid_(x * beta)


class DiffEqWrapper(nn.Module):

    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y):
        if 't' in signature(self.module.forward).parameters:
            return self.module.forward(t, y)
        elif 'y' in signature(self.module.forward).parameters:
            return self.module.forward(y)
        else:
            raise ValueError('Differential equation needs to either take (t, y) or (y,) as input.')

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class MixtureODELayer(nn.Module):
    """Produces a mixture of experts where output = sigma(t) * f(t, x).
    Time-dependent weights sigma(t) help learn to blend the experts without resorting to a highly stiff f.
    Supports both regular and diffeq experts.
    """

    def __init__(self, experts):
        super(MixtureODELayer, self).__init__()
        assert len(experts) > 1
        wrapped_experts = [diffeq_wrapper(ex) for ex in experts]
        self.experts = nn.ModuleList(wrapped_experts)
        self.mixture_weights = nn.Linear(1, len(self.experts))

    def forward(self, t, y):
        dys = []
        for f in self.experts:
            dys.append(f(t, y))
        dys = torch.stack(dys, 0)
        weights = self.mixture_weights(t).view(-1, *([1] * (dys.ndimension() - 1)))
        dy = torch.sum(dys * weights, dim=0, keepdim=False)
        return dy


class Augment(nn.Module):

    def __init__(self, aug_dim):
        super(Augment, self).__init__()
        self.aug_dim = aug_dim

    def forward(self, y, *args, **kwargs):
        z = torch.zeros(y.shape[:-1] + (1 * self.aug_dim,))
        aug_z = torch.cat((y, z), axis=1)
        return aug_z


class DiffEqSequential(DiffEqModule):
    """Entry point for building drift on hidden state of neural network."""

    def __init__(self, *layers: DiffEqModule, explicit_params=True):
        super(DiffEqSequential, self).__init__()
        self.layers = layers if explicit_params else nn.ModuleList(layers)
        self.explicit_params = explicit_params

    def forward(self, t, y, params: Optional[List]=None):
        if params is None:
            for layer in self.layers:
                y = layer(t, y)
        else:
            for layer, params_ in zip(self.layers, params):
                y = layer(t, y, params_)
        return y

    def make_initial_params(self):
        return [layer.make_initial_params() for layer in self.layers]

    def __repr__(self):
        return repr(nn.Sequential(*self.layers)) if self.explicit_params else repr(self.layers)


class ConcatConvTranspose2d(nn.ConvTranspose2d, DiffEqModule):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t=0, output_padding: _size_2_t=0, groups: int=1, bias: bool=True, dilation: int=1, padding_mode: str='zeros'):
        super(ConcatConvTranspose2d, self).__init__(in_channels=in_channels + 1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        self.unpack_params = self.unpack_wb if bias else self.unpack_w

    def forward(self, t, y, params: Optional[List]=None, output_size: Optional[List[int]]=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        weight, bias = self.unpack_params(params)
        ty = utils.channel_cat(t, y)
        output_padding = self._output_padding(ty, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(ty, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    def unpack_wb(self, params: Optional[List]=None):
        if params is None:
            return self.weight, self.bias
        return params

    def unpack_w(self, params: Optional[List]=None):
        if params is None:
            return self.weight, self.bias
        return params[0], None


class Linear(nn.Linear, DiffEqModule):

    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features)

    def forward(self, t, y, params: Optional[List]=None):
        w, b = (self.weight, self.bias) if params is None else params
        return F.linear(y, w, b)


class SqueezeDownsample(DiffEqModule):

    def __init__(self):
        super(SqueezeDownsample, self).__init__()

    def forward(self, t, y, *args, **kwargs):
        del t, args, kwargs
        b, c, h, w = y.size()
        return y.reshape(b, c * 4, h // 2, w // 2)


class ConvDownsample(ConcatConv2d):

    def __init__(self, input_size):
        c, h, w = input_size
        super(ConvDownsample, self).__init__(in_channels=c, out_channels=c * 4, kernel_size=3, stride=2, padding=1)


_shape_t = Union[int, List[int], torch.Size]


class LayerNormalization(nn.LayerNorm, DiffEqModule):

    def __init__(self, normalized_shape: _shape_t, eps: float=1e-05, elementwise_affine: bool=True):
        super(LayerNormalization, self).__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, t, y, params: Optional[List]=None):
        del t
        weight, bias = (self.weight, self.bias) if params is None else params
        return F.layer_norm(y, self.normalized_shape, weight, bias, self.eps)


class Print(DiffEqModule):

    def __init__(self, name=None):
        super(Print, self).__init__()
        self.name = name

    def forward(self, t, y, *args, **kwargs):
        del t, args, kwargs
        msg = f'size: {y.size()}, mean: {y.mean()}, abs mean: {y.abs().mean()}, max: {y.max()}, abs max: {y.abs().max()}, min: {y.min()}, abs min: {y.abs().min()}'
        if self.name is not None:
            msg = f'{self.name}, ' + msg
        logging.warning(msg)
        return y


class YNetWithSplit(nn.Module):

    def __init__(self, *blocks):
        pass

    def forward(self, x):
        zs = []
        net = x
        for block in self.blocks:
            z1, z2 = block(net)
            zs.append(z2)
            net = z1
        return zs


def make_y_net(input_size, blocks=(2, 2, 2), activation='softplus', verbose=False, explicit_params=True, hidden_width=128, aug_dim=0):
    _input_size = (input_size[0] + aug_dim,) + input_size[1:]
    layers = []
    for i, num_blocks in enumerate(blocks, 1):
        for j in range(1, num_blocks + 1):
            layers.extend(diffeq_layers.make_ode_k3_block_layers(input_size=_input_size, activation=activation, last_activation=i < len(blocks) or j < num_blocks, hidden_width=hidden_width))
            if verbose:
                if i == 1:
                    None
                layers.append(diffeq_layers.Print(name=f'group: {i}, block: {j}'))
        if i < len(blocks):
            layers.append(diffeq_layers.ConvDownsample(_input_size))
            _input_size = _input_size[0] * 4, _input_size[1] // 2, _input_size[2] // 2
    y_net = diffeq_layers.DiffEqSequential(*layers, explicit_params=explicit_params)
    return y_net, _input_size


class BaselineYNet(nn.Module):

    def __init__(self, input_size=(3, 32, 32), num_classes=10, activation='softplus', residual=False, hidden_width=128, aug=0):
        super(BaselineYNet, self).__init__()
        y_net, output_size = make_y_net(input_size=input_size, explicit_params=False, activation=activation, hidden_width=hidden_width)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(output_size)) + aug, num_classes))
        self.y_net = y_net
        self.residual = residual

    def forward(self, y, *args, **kwargs):
        t = y.new_tensor(0.0)
        outs = self.y_net(t, y).flatten(start_dim=1)
        if self.residual:
            outs += y.flatten(start_dim=1)
        return self.projection(outs), torch.tensor(0.0, device=y.device)


NGROUPS = 16


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim, conv_block=None):
        super(BasicBlock, self).__init__()
        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d
        self.norm1 = nn.GroupNorm(NGROUPS, dim, eps=0.0001)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(NGROUPS, dim, eps=0.0001)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)

    def forward(self, t, x):
        residual = x
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)
        out += residual
        return out


class ResNet(container.SequentialDiffEq):

    def __init__(self, dim, intermediate_dim, n_resblocks, conv_block=None):
        super(ResNet, self).__init__()
        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks
        layers = []
        layers.append(conv_block(dim, intermediate_dim, ksize=3, stride=1, padding=1, bias=False))
        for _ in range(n_resblocks):
            layers.append(BasicBlock(intermediate_dim, conv_block))
        layers.append(nn.GroupNorm(NGROUPS, intermediate_dim, eps=0.0001))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_block(intermediate_dim, dim, ksize=1, bias=False))
        super(ResNet, self).__init__(*layers)

    def __repr__(self):
        return '{name}({dim}, intermediate_dim={intermediate_dim}, n_resblocks={n_resblocks})'.format(name=self.__class__.__name__, **self.__dict__)


@torch.jit.script
def _swish(x, beta):
    return x * torch.sigmoid(x * F.softplus(beta))


class Swish(nn.Module):

    def __init__(self, beta=0.5):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return _swish(x, beta=self.beta)

    def _forward_unimplemented(self, *input: Any) ->None:
        pass


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):

    def forward(self, x):
        return _mish(x)


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.view((-1,) + self.shape)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.get_default_dtype()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        offset = self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0), :]
        x = x + offset
        return self.dropout(x)


class OptimizedModel(abc.ABC, nn.Module):

    def zero_grad(self) ->None:
        for p in self.parameters():
            p.grad = None


class VerboseSequential(OptimizedModel):

    def __init__(self, *args, verbose=False, stream: str='stdout'):
        super(VerboseSequential, self).__init__()
        self.layers = nn.ModuleList(args)
        self.forward = self._forward_verbose if verbose else self._forward
        self.stream = stream

    def _forward_verbose(self, net):
        stream = {'stdout': sys.stdout, 'stderr': sys.stderr}[self.stream] if self.stream in ('stdout', 'stderr') else self.stream
        None
        for i, layer in enumerate(self.layers):
            net = layer(net)
            None
        return net

    def _forward(self, net):
        for layer in self.layers:
            net = layer(net)
        return net


class Module(nn.Module):

    def zero_grad(self) ->None:
        for p in self.parameters():
            p.grad = None

    @property
    def device(self):
        return next(self.parameters()).device


class ReshapeDiffEq(nn.Module):

    def __init__(self, input_shape, net):
        super(ReshapeDiffEq, self).__init__()
        assert len(signature(net.forward).parameters) == 2, 'use diffeq_wrapper before reshape_wrapper.'
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Augment,
     lambda: ([], {'aug_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BlendConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlendLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatCoordConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([1, 1, 1, 1]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatLinear_v2,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 256, 4]), torch.rand([4, 4, 16384, 4])], {}),
     True),
    (ConcatSquashConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([1, 1]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiffEqSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedConvTranspose,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HyperConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([1, 1]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HyperLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([1, 1]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IgnoreConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IgnoreLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormalization,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Print,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequentialDiffEq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SquashLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 256, 4]), torch.rand([4, 4, 16384, 4])], {}),
     True),
    (SqueezeDownsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeDependentSwish,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 256, 4])], {}),
     True),
    (VerboseSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (View,
     lambda: ([], {'shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xwinxu_bayeSDE(_paritybench_base):
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

