import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
brownian = _module
profile_btree = _module
diagnostics = _module
inspection = _module
ito_additive = _module
ito_diagonal = _module
ito_general = _module
ito_scalar = _module
run_all = _module
stratonovich_additive = _module
stratonovich_diagonal = _module
stratonovich_general = _module
stratonovich_scalar = _module
utils = _module
examples = _module
cont_ddpm = _module
latent_sde = _module
latent_sde_lorenz = _module
sde_gan = _module
unet = _module
setup = _module
tests = _module
problems = _module
test_adjoint = _module
test_brownian_interval = _module
test_brownian_path = _module
test_brownian_tree = _module
test_sdeint = _module
utils = _module
torchsde = _module
_brownian = _module
brownian_base = _module
brownian_interval = _module
derived = _module
_core = _module
adaptive_stepping = _module
adjoint = _module
adjoint_sde = _module
base_sde = _module
base_solver = _module
better_abc = _module
interp = _module
methods = _module
euler = _module
euler_heun = _module
heun = _module
log_ode = _module
midpoint = _module
milstein = _module
reversible_heun = _module
srk = _module
tableaus = _module
sra1 = _module
sra2 = _module
sra3 = _module
srid1 = _module
srid2 = _module
misc = _module
sdeint = _module
settings = _module
types = _module

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


import logging


import time


import numpy.random as npr


import torch


import matplotlib.pyplot as plt


import itertools


import random


import numpy as np


from scipy import stats


import abc


import math


import torchvision as tv


from torch import nn


from torch import optim


from torch.utils import data


from collections import namedtuple


from typing import Optional


from typing import Union


from torch import distributions


from typing import Sequence


import matplotlib.gridspec as gridspec


from torch.distributions import Normal


import torch.optim.swa_utils as swa_utils


import torch.nn.functional as F


import re


from scipy.stats import kstest


from scipy.stats import norm


import copy


import warnings


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Callable


Module = torch.nn.Module


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


class ScoreMatchingSDE(Module):
    """Wraps score network with analytical sampling and cond. score computation.

    The variance preserving formulation in
        Score-Based Generative Modeling through Stochastic Differential Equations
        https://arxiv.org/abs/2011.13456
    """

    def __init__(self, denoiser, input_size=(1, 28, 28), t0=0.0, t1=1.0, beta_min=0.1, beta_max=20.0):
        super(ScoreMatchingSDE, self).__init__()
        if t0 > t1:
            raise ValueError(f'Expected t0 <= t1, but found t0={t0:.4f}, t1={t1:.4f}')
        self.input_size = input_size
        self.denoiser = denoiser
        self.t0 = t0
        self.t1 = t1
        self.beta_min = beta_min
        self.beta_max = beta_max

    def score(self, t, y):
        if isinstance(t, float):
            t = y.new_tensor(t)
        if t.dim() == 0:
            t = t.repeat(y.shape[0])
        return self.denoiser(t, y)

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _indefinite_int(self, t):
        """Indefinite integral of beta(t)."""
        return self.beta_min * t + 0.5 * t ** 2 * (self.beta_max - self.beta_min)

    def analytical_mean(self, t, x_t0):
        mean_coeff = (-0.5 * (self._indefinite_int(t) - self._indefinite_int(self.t0))).exp()
        mean = x_t0 * fill_tail_dims(mean_coeff, x_t0)
        return mean

    def analytical_var(self, t, x_t0):
        analytical_var = 1 - (-self._indefinite_int(t) + self._indefinite_int(self.t0)).exp()
        return analytical_var

    @torch.no_grad()
    def analytical_sample(self, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(self, x_t, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return -(x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-05)

    def f(self, t, y):
        return -0.5 * self._beta(t) * y

    def g(self, t, y):
        return fill_tail_dims(self._beta(t).sqrt(), y).expand_as(y)

    def sample_t1_marginal(self, batch_size, tau=1.0):
        return torch.randn(size=(batch_size, *self.input_size), device=self.device) * math.sqrt(tau)

    def lambda_t(self, t):
        return self.analytical_var(t, None)

    def forward(self, x_t0, partitions=1):
        """Compute the score matching objective.
        Split [t0, t1] into partitions; sample uniformly on each partition to reduce gradient variance.
        """
        u = torch.rand(size=(x_t0.shape[0], partitions), dtype=x_t0.dtype, device=x_t0.device)
        u.mul_((self.t1 - self.t0) / partitions)
        shifts = torch.arange(0, partitions, device=x_t0.device, dtype=x_t0.dtype)[None, :]
        shifts.mul_((self.t1 - self.t0) / partitions).add_(self.t0)
        t = (u + shifts).reshape(-1)
        lambda_t = self.lambda_t(t)
        x_t0 = x_t0.repeat_interleave(partitions, dim=0)
        x_t = self.analytical_sample(t, x_t0)
        fake_score = self.score(t, x_t)
        true_score = self.analytical_score(x_t, t, x_t0)
        loss = lambda_t * ((fake_score - true_score) ** 2).flatten(start_dim=1).sum(dim=1)
        return loss


class ReverseDiffeqWrapper(Module):
    """Wrapper of the score network for odeint/sdeint.

    We split this module out, so that `forward` of the score network is solely
    used for computing the score, and the `forward` here is used for odeint.
    Helps with data parallel.
    """
    noise_type = 'diagonal'
    sde_type = 'stratonovich'

    def __init__(self, module: ScoreMatchingSDE):
        super(ReverseDiffeqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y):
        return -(self.module.f(-t, y) - 0.5 * self.module.g(-t, y) ** 2 * self.module.score(-t, y))

    def f(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -(self.module.f(-t, y) - self.module.g(-t, y) ** 2 * self.module.score(-t, y))
        return out.flatten(start_dim=1)

    def g(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -self.module.g(-t, y)
        return out.flatten(start_dim=1)

    def sample_t1_marginal(self, batch_size, tau=1.0):
        return self.module.sample_t1_marginal(batch_size, tau)

    @torch.no_grad()
    def ode_sample(self, batch_size=64, tau=1.0, t=None, y=None, dt=0.01):
        self.module.eval()
        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y
        return torchdiffeq.odeint(self, y, t, method='rk4', options={'step_size': dt})

    @torch.no_grad()
    def ode_sample_final(self, batch_size=64, tau=1.0, t=None, y=None, dt=0.01):
        return self.ode_sample(batch_size, tau, t, y, dt)[-1]

    @torch.no_grad()
    def sde_sample(self, batch_size=64, tau=1.0, t=None, y=None, dt=0.01, tweedie_correction=True):
        self.module.eval()
        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y
        ys = torchsde.sdeint(self, y.flatten(start_dim=1), t, dt=dt)
        ys = ys.view(len(t), *y.size())
        if tweedie_correction:
            ys[-1] = self.tweedie_correction(self.t0, ys[-1], dt)
        return ys

    @torch.no_grad()
    def sde_sample_final(self, batch_size=64, tau=1.0, t=None, y=None, dt=0.01):
        return self.sde_sample(batch_size, tau, t, y, dt)[-1]

    def tweedie_correction(self, t, y, dt):
        return y + dt ** 2 * self.module.score(t, y)

    @property
    def t0(self):
        return self.module.t0

    @property
    def t1(self):
        return self.module.t1


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        self.f_net = nn.Sequential(nn.Linear(latent_size + context_size, hidden_size), nn.Softplus(), nn.Linear(hidden_size, hidden_size), nn.Softplus(), nn.Linear(hidden_size, latent_size))
        self.h_net = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.Softplus(), nn.Linear(hidden_size, hidden_size), nn.Softplus(), nn.Linear(hidden_size, latent_size))
        self.g_nets = nn.ModuleList([nn.Sequential(nn.Linear(1, hidden_size), nn.Softplus(), nn.Linear(hidden_size, 1), nn.Sigmoid()) for _ in range(latent_size)])
        self.projector = nn.Linear(latent_size, data_size)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for g_net_i, y_i in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method='euler'):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        if adjoint:
            adjoint_params = (ctx,) + tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            zs, log_ratio = torchsde.sdeint_adjoint(self, z0, ts, adjoint_params=adjoint_params, dt=0.01, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=0.01, logqp=True, method=method)
        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=0.001, bm=bm)
        _xs = self.projector(zs)
        return _xs


class LipSwish(torch.nn.Module):

    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):

    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()
        model = [torch.nn.Linear(in_size, mlp_size), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


class Generator(torch.nn.Module):

    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ts, batch_size):
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0, adjoint_method='adjoint_reversible_heun')
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


class DiscriminatorFunc(torch.nn.Module):

    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):

    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0, adjoint_method='adjoint_reversible_heun', adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):

    def forward(self, x):
        return _mish(x)


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SelfAttention(nn.Module):

    def __init__(self, dim, groups=32, **kwargs):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.group_norm(x)
        q, k, v = tuple(t.view(b, c, h * w) for t in self.qkv(x).chunk(chunks=3, dim=1))
        attn_matrix = (torch.bmm(k.permute(0, 2, 1), q) / math.sqrt(c)).softmax(dim=-2)
        out = torch.bmm(v, attn_matrix).view(b, c, h, w)
        return self.out(out)


class LinearTimeSelfAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8, dropout_rate=0.0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.groups = groups
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = nn.Sequential(nn.GroupNorm(groups, dim), Mish(), nn.Conv2d(dim, dim_out, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(groups, dim_out), Mish(), nn.Dropout(p=dropout_rate), nn.Conv2d(dim_out, dim_out, 3, padding=1))
        self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x, t):
        h = self.block1(x)
        h += self.mlp(t)[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim}, dim_out={self.dim_out}, time_emb_dim={self.time_emb_dim}, groups={self.groups}, dropout_rate={self.dropout_rate})'


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Blur(nn.Module):

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        f = f[None, None, :] * f[None, :, None]
        self.register_buffer('f', f)

    def forward(self, x):
        return kornia.filter2D(x, self.f, normalized=True)


class Downsample(nn.Module):

    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.conv = nn.Sequential(Blur(), nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1))
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.conv = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1), Blur())
        else:
            self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, input_size=(3, 32, 32), hidden_channels=64, dim_mults=(1, 2, 4, 8), groups=32, heads=4, dim_head=32, dropout_rate=0.0, num_res_blocks=2, attn_resolutions=(16,), attention_cls=SelfAttention):
        super().__init__()
        in_channels, in_height, in_width = input_size
        dims = [hidden_channels, *map(lambda m: hidden_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_pos_emb = SinusoidalPosEmb(hidden_channels)
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), Mish(), nn.Linear(hidden_channels * 4, hidden_channels))
        self.first_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        h, w = in_height, in_width
        self.down_res_blocks = nn.ModuleList([])
        self.down_attn_blocks = nn.ModuleList([])
        self.down_spatial_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            res_blocks = nn.ModuleList([ResnetBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate)])
            res_blocks.extend([ResnetBlock(dim=dim_out, dim_out=dim_out, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate) for _ in range(num_res_blocks - 1)])
            self.down_res_blocks.append(res_blocks)
            attn_blocks = nn.ModuleList([])
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.extend([Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups)) for _ in range(num_res_blocks)])
            self.down_attn_blocks.append(attn_blocks)
            if ind < len(in_out) - 1:
                spatial_blocks = nn.ModuleList([Downsample(dim_out)])
                h, w = h // 2, w // 2
            else:
                spatial_blocks = nn.ModuleList()
            self.down_spatial_blocks.append(spatial_blocks)
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate)
        self.mid_attn = Residual(attention_cls(mid_dim, heads=heads, dim_head=dim_head, groups=groups))
        self.mid_block2 = ResnetBlock(dim=mid_dim, dim_out=mid_dim, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate)
        self.ups_res_blocks = nn.ModuleList([])
        self.ups_attn_blocks = nn.ModuleList([])
        self.ups_spatial_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            res_blocks = nn.ModuleList([ResnetBlock(dim=dim_out * 2, dim_out=dim_out, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate) for _ in range(num_res_blocks)])
            res_blocks.extend([ResnetBlock(dim=dim_out + dim_in, dim_out=dim_in, time_emb_dim=hidden_channels, groups=groups, dropout_rate=dropout_rate)])
            self.ups_res_blocks.append(res_blocks)
            attn_blocks = nn.ModuleList([])
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.extend([Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups)) for _ in range(num_res_blocks)])
                attn_blocks.append(Residual(attention_cls(dim_in, heads=heads, dim_head=dim_head, groups=groups)))
            self.ups_attn_blocks.append(attn_blocks)
            spatial_blocks = nn.ModuleList()
            if ind < len(in_out) - 1:
                spatial_blocks.append(Upsample(dim_in))
                h, w = h * 2, w * 2
            self.ups_spatial_blocks.append(spatial_blocks)
        self.final_conv = nn.Sequential(nn.GroupNorm(groups, hidden_channels), Mish(), nn.Conv2d(hidden_channels, in_channels, 1))

    def forward(self, t, x):
        t = self.mlp(self.time_pos_emb(t))
        hs = [self.first_conv(x)]
        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(zip(self.down_res_blocks, self.down_attn_blocks, self.down_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    h = res_block(hs[-1], t)
                    h = attn_block(h)
                    hs.append(h)
            else:
                for res_block in res_blocks:
                    h = res_block(hs[-1], t)
                    hs.append(h)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                hs.append(spatial_block(hs[-1]))
        h = hs[-1]
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(zip(self.ups_res_blocks, self.ups_attn_blocks, self.ups_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    h = res_block(torch.cat((h, hs.pop()), dim=1), t)
                    h = attn_block(h)
            else:
                for res_block in res_blocks:
                    h = res_block(torch.cat((h, hs.pop()), dim=1), t)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                h = spatial_block(h)
        return self.final_conv(h)


class FGSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FGSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def g(self, t, y):
        return y.unsqueeze(-1).sigmoid() * self.vector


class FAndGSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector


class GProdSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(GProdSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGProdSDE(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGProdSDE, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g_prod(self, t, y, v):
        return -y, (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGGProdSDE1(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGGProdSDE1, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class FAndGGProdSDE2(torch.nn.Module):
    noise_type = 'general'

    def __init__(self, sde_type, vector):
        super(FAndGGProdSDE2, self).__init__()
        self.sde_type = sde_type
        self.register_buffer('vector', vector)

    def f(self, t, y):
        return -y

    def f_and_g(self, t, y):
        return -y, y.unsqueeze(-1).sigmoid() * self.vector

    def g_prod(self, t, y, v):
        return (y.unsqueeze(-1).sigmoid() * self.vector).bmm(v.unsqueeze(-1)).squeeze(-1)


class ContainerMeta(type):

    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith('__'))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()


class NOISE_TYPES(metaclass=ContainerMeta):
    general = 'general'
    diagonal = 'diagonal'
    scalar = 'scalar'
    additive = 'additive'


class SDE_TYPES(metaclass=ContainerMeta):
    ito = 'ito'
    stratonovich = 'stratonovich'


class BaseSDE(abc.ABC, nn.Module):
    """Base class for all SDEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        if noise_type not in NOISE_TYPES:
            raise ValueError(f'Expected noise type in {NOISE_TYPES}, but found {noise_type}')
        if sde_type not in SDE_TYPES:
            raise ValueError(f'Expected sde type in {SDE_TYPES}, but found {sde_type}')
        self.noise_type = noise_type
        self.sde_type = sde_type


class ForwardSDE(BaseSDE):

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde
        if hasattr(sde, 'f_and_g_prod'):
            self.f_and_g_prod = sde.f_and_g_prod
        elif hasattr(sde, 'f') and hasattr(sde, 'g_prod'):
            self.f_and_g_prod = self.f_and_g_prod_default1
        else:
            self.f_and_g_prod = self.f_and_g_prod_default2
        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {NOISE_TYPES.diagonal: self.prod_diagonal}.get(sde.noise_type, self.prod_default)
        self.g_prod_and_gdg_prod = {NOISE_TYPES.diagonal: self.g_prod_and_gdg_prod_diagonal, NOISE_TYPES.additive: self.g_prod_and_gdg_prod_additive}.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {NOISE_TYPES.general: self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1}.get(sde.noise_type, self._return_zero)

    def f_default(self, t, y):
        raise RuntimeError('Method `f` has not been provided, but is required for this method.')

    def g_default(self, t, y):
        raise RuntimeError('Method `g` has not been provided, but is required for this method.')

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, v):
        f, g = self.f_and_g(t, y)
        return f, self.prod(g, v)

    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(outputs=g, inputs=y, grad_outputs=g * v2.unsqueeze(-2), retain_graph=True, create_graph=requires_grad, allow_unused=True)
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(outputs=g, inputs=y, grad_outputs=g * v2, retain_graph=True, create_graph=requires_grad, allow_unused=True)
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.0

    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [misc.jvp(outputs=g[..., col_idx], inputs=y, grad_inputs=ga[..., col_idx], retain_graph=True, create_graph=requires_grad, allow_unused=True)[0] for col_idx in range(g.size(-1))]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = misc.jvp(outputs=g_dup, inputs=y_dup, grad_inputs=ga_flat, create_graph=requires_grad, allow_unused=True)
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):
        return 0.0


class RenameMethodsSDE(BaseSDE):

    def __init__(self, sde, drift='f', diffusion='g', prior_drift='h', diffusion_prod='g_prod', drift_and_diffusion='f_and_g', drift_and_diffusion_prod='f_and_g_prod'):
        super(RenameMethodsSDE, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        for name, value in zip(('f', 'g', 'h', 'g_prod', 'f_and_g', 'f_and_g_prod'), (drift, diffusion, prior_drift, diffusion_prod, drift_and_diffusion, drift_and_diffusion_prod)):
            try:
                setattr(self, name, getattr(sde, value))
            except AttributeError:
                pass


class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.ito)


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.stratonovich)


class SDELogqp(BaseSDE):

    def __init__(self, sde):
        super(SDELogqp, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        try:
            self._base_f = sde.f
            self._base_g = sde.g
            self._base_h = sde.h
        except AttributeError as e:
            raise AttributeError('If using logqp then drift, diffusion and prior drift must all be specified.') from e
        if sde.noise_type == NOISE_TYPES.diagonal:
            self.f = self.f_diagonal
            self.g = self.g_diagonal
            self.f_and_g = self.f_and_g_diagonal
        else:
            self.f = self.f_general
            self.g = self.g_general
            self.f_and_g = self.f_and_g_general

    def f_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        g = self._base_g(t, y)
        g_logqp = y.new_zeros(size=(y.size(0), 1))
        return torch.cat([g, g_logqp], dim=1)

    def f_and_g_diagonal(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        g_logqp = y.new_zeros(size=(y.size(0), 1))
        return torch.cat([f, f_logqp], dim=1), torch.cat([g, g_logqp], dim=1)

    def f_general(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.batch_mvp(g.pinverse(), f - h)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_general(self, t, y: Tensor):
        y = y[:, :-1]
        g = self._base_sde.g(t, y)
        g_logqp = y.new_zeros(size=(g.size(0), 1, g.size(-1)))
        return torch.cat([g, g_logqp], dim=1)

    def f_and_g_general(self, t, y: Tensor):
        y = y[:, :-1]
        f, g, h = self._base_f(t, y), self._base_g(t, y), self._base_h(t, y)
        u = misc.batch_mvp(g.pinverse(), f - h)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        g_logqp = y.new_zeros(size=(g.size(0), 1, g.size(-1)))
        return torch.cat([f, f_logqp], dim=1), torch.cat([g, g_logqp], dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Encoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (LipSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'mlp_size': 4, 'num_layers': 1, 'tanh': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_google_research_torchsde(_paritybench_base):
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

