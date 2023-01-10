import sys
_module = sys.modules[__name__]
del sys
denoising_diffusion_pytorch = _module
classifier_free_guidance = _module
continuous_time_gaussian_diffusion = _module
denoising_diffusion_pytorch = _module
denoising_diffusion_pytorch_1d = _module
elucidated_diffusion = _module
learned_gaussian_diffusion = _module
v_param_continuous_time_gaussian_diffusion = _module
version = _module
weighted_objective_gaussian_diffusion = _module
setup = _module

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


import math


import copy


from random import random


from functools import partial


from collections import namedtuple


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


from torch import sqrt


from torch.special import expm1


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim import Adam


from torchvision import transforms as T


from torchvision import utils


from math import sqrt


from math import pi


from math import log as ln


from inspect import isfunction


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def exists(x):
    return x is not None


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(dim, default(dim_out, dim), 3, padding=1))


class Unet(nn.Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False, resnet_block_groups=8, learned_variance=False, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim), block_klass(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_variance'])


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-05):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class GaussianDiffusion(nn.Module):

    def __init__(self, model, *, image_size, timesteps=1000, sampling_timesteps=None, loss_type='l1', objective='pred_noise', beta_schedule='cosine', schedule_fn_kwargs=dict(), p2_loss_weight_gamma=0.0, p2_loss_weight_k=1, ddim_sampling_eta=0.0):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda : torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        model_out = self.model(x, t, x_self_cond)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class MonotonicLinear(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())


class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(self, *, log_snr_max, log_snr_min, hidden_dim=1024, frac_gradient=1.0):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max
        self.net = nn.Sequential(Rearrange('... -> ... 1'), MonotonicLinear(1, 1), Residual(nn.Sequential(MonotonicLinear(1, hidden_dim), nn.Sigmoid(), MonotonicLinear(hidden_dim, 1))), Rearrange('... 1 -> ...'))
        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device
        out_zero = self.net(torch.zeros_like(x))
        out_one = self.net(torch.ones_like(x))
        x = self.net(x)
        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def alpha_cosine_log_snr(t, s=0.008):
    return -log(torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2 - 1, eps=1e-05)


def beta_linear_log_snr(t):
    return -log(expm1(0.0001 + 10 * t ** 2))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class ContinuousTimeGaussianDiffusion(nn.Module):

    def __init__(self, model, *, image_size, channels=3, loss_type='l1', noise_schedule='linear', num_sample_steps=500, clip_sample_denoised=True, learned_schedule_net_hidden_dim=1024, learned_noise_schedule_frac_gradient=1.0, p2_loss_weight_gamma=0.0, p2_loss_weight_k=1):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.loss_type = loss_type
        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0.0, 1.0)]
            self.log_snr = learned_noise_schedule(log_snr_max=log_snr_max, log_snr_min=log_snr_min, hidden_dim=learned_schedule_net_hidden_dim, frac_gradient=learned_noise_schedule_frac_gradient)
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        assert p2_loss_weight_gamma <= 2, 'in paper, they noticed any gamma greater than 2 is harmful'
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))
        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_noise = self.model(x, batch_log_snr)
        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha
            x_start.clamp_(-1.0, 1.0)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr)
        losses = self.loss_fn(model_out, noise, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        if self.p2_loss_weight_gamma >= 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -self.p2_loss_weight_gamma
            losses = losses * loss_weight
        return losses.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)


class Unet1D(nn.Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False, resnet_block_groups=8, learned_variance=False, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim), block_klass(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class GaussianDiffusion1D(nn.Module):

    def __init__(self, model, *, seq_length, timesteps=1000, sampling_timesteps=None, loss_type='l1', objective='pred_noise', beta_schedule='sigmoid', p2_loss_weight_gamma=0.0, p2_loss_weight_k=1, ddim_sampling_eta=0.0):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.seq_length = seq_length
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda : torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        model_out = self.model(x, t, x_self_cond)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class ElucidatedDiffusion(nn.Module):

    def __init__(self, net, *, image_size, channels=3, num_sample_steps=32, sigma_min=0.002, sigma_max=80, sigma_data=0.5, rho=7, P_mean=-1.2, P_std=1.2, S_churn=80, S_tmin=0.05, S_tmax=50, S_noise=1.003):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition
        self.net = net
        self.channels = channels
        self.image_size = image_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False):
        batch, device = noised_images.shape[0], noised_images.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)
        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
        net_out = self.net(self.c_in(padded_sigma) * noised_images, self.c_noise(sigma), self_cond)
        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        N = num_sample_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size=16, num_sample_steps=None, clamp=True):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = batch_size, self.channels, self.image_size, self.image_size
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where((sigmas >= self.S_tmin) & (sigmas <= self.S_tmax), min(self.S_churn / num_sample_steps, sqrt(2) - 1), 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        init_sigma = sigmas[0]
        images = init_sigma * torch.randn(shape, device=self.device)
        x_start = None
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))
            eps = self.S_noise * torch.randn(shape, device=self.device)
            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps
            self_cond = x_start if self.self_condition else None
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp=clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat
            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp=clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)
            images = images_next
            x_start = model_output
        images = images.clamp(-1.0, 1.0)
        return unnormalize_to_zero_to_one(images)

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, images):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels
        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'
        images = normalize_to_neg_one_to_one(images)
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')
        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise
        self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images, sigmas)
                self_cond.detach_()
        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)
        losses = F.mse_loss(denoised, images, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


NAT = 1.0 / ln(2)


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -thres, log_cdf_plus, torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta)))
    return log_probs


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


class LearnedGaussianDiffusion(GaussianDiffusion):

    def __init__(self, model, vb_loss_weight=0.001, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        assert model.out_dim == model.channels * 2, 'dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`'
        assert not model.self_condition, 'not supported yet'
        self.vb_loss_weight = vb_loss_weight

    def model_predictions(self, x, t):
        model_output = self.model(x, t)
        model_output, pred_variance = model_output.chunk(2, dim=1)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
        return ModelPrediction(pred_noise, x_start, pred_variance)

    def p_mean_variance(self, *, x, t, clip_denoised, model_output=None):
        model_output = default(model_output, lambda : self.model(x, t))
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        var_interp_frac = unnormalize_to_zero_to_one(var_interp_frac_unnormalized)
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, _, _ = self.q_posterior(x_start, x, t)
        return model_mean, model_variance, model_log_variance

    def p_losses(self, x_start, t, noise=None, clip_denoised=False):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_t, t)
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x_t, t=t, clip_denoised=clip_denoised, model_output=model_output)
        detached_model_mean = model_mean.detach()
        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=detached_model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT
        vb_losses = torch.where(t == 0, decoder_nll, kl)
        pred_noise, _ = model_output.chunk(2, dim=1)
        simple_losses = self.loss_fn(pred_noise, noise)
        return simple_losses + vb_losses.mean() * self.vb_loss_weight


class VParamContinuousTimeGaussianDiffusion(nn.Module):
    """
    a new type of parameterization in v-space proposed in https://arxiv.org/abs/2202.00512 that
    (1) allows for improved distillation over noise prediction objective and
    (2) noted in imagen-video to improve upsampling unets by removing the color shifting artifacts
    """

    def __init__(self, model, *, image_size, channels=3, num_sample_steps=500, clip_sample_denoised=True):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.log_snr = alpha_cosine_log_snr
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))
        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_v = self.model(x, batch_log_snr)
        x_start = alpha * x - sigma * pred_v
        if self.clip_sample_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr, alpha, sigma

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x, log_snr, alpha, sigma = self.q_sample(x_start=x_start, times=times, noise=noise)
        v = alpha * noise - sigma * x_start
        model_out = self.model(x, log_snr)
        return F.mse_loss(model_out, v)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)


class WeightedObjectiveGaussianDiffusion(GaussianDiffusion):

    def __init__(self, model, *args, pred_noise_loss_weight=0.1, pred_x_start_loss_weight=0.1, **kwargs):
        super().__init__(model, *args, **kwargs)
        channels = model.channels
        assert model.out_dim == channels * 2 + 2, 'dimension out (out_dim) of unet must be twice the number of channels + 2 (for the softmax weighted sum) - for channels of 3, this should be (3 * 2) + 2 = 8'
        assert not model.self_condition, 'not supported yet'
        assert not self.is_ddim_sampling, 'ddim sampling cannot be used'
        self.split_dims = channels, channels, 2
        self.pred_noise_loss_weight = pred_noise_loss_weight
        self.pred_x_start_loss_weight = pred_x_start_loss_weight

    def p_mean_variance(self, *, x, t, clip_denoised, model_output=None):
        model_output = self.model(x, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim=1)
        normalized_weights = weights.softmax(dim=1)
        x_start_from_noise = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        x_starts = torch.stack((x_start_from_noise, pred_x_start), dim=1)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', normalized_weights, x_starts)
        if clip_denoised:
            weighted_x_start.clamp_(-1.0, 1.0)
        model_mean, model_variance, model_log_variance = self.q_posterior(weighted_x_start, x, t)
        return model_mean, model_variance, model_log_variance

    def p_losses(self, x_start, t, noise=None, clip_denoised=False):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_t, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim=1)
        noise_loss = self.loss_fn(noise, pred_noise) * self.pred_noise_loss_weight
        x_start_loss = self.loss_fn(x_start, pred_x_start) * self.pred_x_start_loss_weight
        x_start_from_pred_noise = self.predict_start_from_noise(x_t, t, pred_noise)
        x_start_from_pred_noise = x_start_from_pred_noise.clamp(-2.0, 2.0)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', weights.softmax(dim=1), torch.stack((x_start_from_pred_noise, pred_x_start), dim=1))
        weighted_x_start_loss = self.loss_fn(x_start, weighted_x_start)
        return weighted_x_start_loss + x_start_loss + noise_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MonotonicLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_denoising_diffusion_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

