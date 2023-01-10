import sys
_module = sys.modules[__name__]
del sys
audio_diffusion_pytorch = _module
diffusion = _module
model = _module
modules = _module
utils = _module
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


from math import atan


from math import cos


from math import pi


from math import sin


from math import sqrt


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


from typing import Tuple


from typing import Type


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from random import randint


from typing import Sequence


from typing import Union


from torch import nn


from math import floor


from math import log


from torch import einsum


from functools import reduce


from inspect import isfunction


from math import ceil


from math import log2


from typing import Dict


from typing import TypeVar


class Diffusion(nn.Module):
    alias: str = ''
    """Base diffusion class"""

    def denoise_fn(self, x_noisy: Tensor, sigmas: Optional[Tensor]=None, sigma: Optional[float]=None, **kwargs) ->Tensor:
        raise NotImplementedError('Diffusion class missing denoise_fn')

    def forward(self, x: Tensor, noise: Tensor=None, **kwargs) ->Tensor:
        raise NotImplementedError('Diffusion class missing forward function')


class Distribution:

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


T = TypeVar('T')


def default(val: Optional[T], d: Union[Callable[..., T], T]) ->T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_batch(batch_size: int, device: torch.device, x: Optional[float]=None, xs: Optional[Tensor]=None) ->Tensor:
    assert exists(x) ^ exists(xs), 'Either x or xs must be provided'
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x)
    assert exists(xs)
    return xs


class VDiffusion(Diffusion):
    alias = 'v'

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(self, x_noisy: Tensor, sigmas: Optional[Tensor]=None, sigma: Optional[float]=None, **kwargs) ->Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Tensor=None, **kwargs) ->Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, 'b -> b 1 1')
        noise = default(noise, lambda : torch.randn_like(x))
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        return F.mse_loss(x_denoised, x_target)


def pad_dims(x: Tensor, ndim: int) ->Tensor:
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float=0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        x_flat = rearrange(x, 'b ... -> b (...)')
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        scale.clamp_(min=1.0)
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""
    alias = 'k'

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution, sigma_data: float, dynamic_threshold: float=0.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) ->Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, 'b -> b 1 1')
        c_skip = sigma_data ** 2 / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(self, x_noisy: Tensor, sigmas: Optional[Tensor]=None, sigma: Optional[float]=None, **kwargs) ->Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)

    def loss_weight(self, sigmas: Tensor) ->Tensor:
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor=None, **kwargs) ->Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, 'b -> b 1 1')
        noise = default(noise, lambda : torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)
        losses = F.mse_loss(x_denoised, x, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss


class VKDiffusion(Diffusion):
    alias = 'vk'

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) ->Tuple[Tensor, ...]:
        sigma_data = 1.0
        sigmas = rearrange(sigmas, 'b -> b 1 1')
        c_skip = sigma_data ** 2 / (sigmas ** 2 + sigma_data ** 2)
        c_out = -sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) ->Tensor:
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) ->Tensor:
        return (t * pi / 2).tan()

    def denoise_fn(self, x_noisy: Tensor, sigmas: Optional[Tensor]=None, sigma: Optional[float]=None, **kwargs) ->Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor=None, **kwargs) ->Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, 'b -> b 1 1')
        noise = default(noise, lambda : torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-07)
        loss = F.mse_loss(x_pred, v_target)
        return loss


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) ->Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):

    def forward(self, num_steps: int, device: Any) ->Tensor:
        sigmas = torch.linspace(1, 0, num_steps + 1)[:-1]
        return sigmas


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float=7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) ->Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (self.sigma_max ** rho_inv + steps / (num_steps - 1) * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


class Sampler(nn.Module):
    diffusion_types: List[Type[Diffusion]] = []

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) ->Tensor:
        raise NotImplementedError()

    def inpaint(self, source: Tensor, mask: Tensor, fn: Callable, sigmas: Tensor, num_steps: int, num_resamples: int) ->Tensor:
        raise NotImplementedError('Inpainting not available with current sampler')


class VSampler(Sampler):
    diffusion_types = [VDiffusion]

    def get_alpha_beta(self, sigma: float) ->Tuple[float, float]:
        angle = sigma * pi / 2
        alpha = cos(angle)
        beta = sin(angle)
        return alpha, beta

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) ->Tensor:
        x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[0].item())
        for i in range(num_steps - 1):
            is_last = i == num_steps - 1
            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha
            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1].item())
                x = x_pred * alpha + x_eps * beta
        return x_pred


class KarrasSampler(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 1"""
    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, s_tmin: float=0, s_tmax: float=float('inf'), s_churn: float=0.0, s_noise: float=1.0):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float, gamma: float) ->Tensor:
        """Algorithm 2 (step)"""
        sigma_hat = sigma + gamma * sigma
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) ->Tensor:
        x = sigmas[0] * noise
        gammas = torch.where((sigmas >= self.s_tmin) & (sigmas <= self.s_tmax), min(self.s_churn / num_steps, sqrt(2) - 1), 0.0)
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i])
        return x


class AEulerSampler(Sampler):
    diffusion_types = [KDiffusion, VKDiffusion]

    def get_sigmas(self, sigma: float, sigma_next: float) ->Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) ->Tensor:
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_next = x + d * (sigma_down - sigma)
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) ->Tensor:
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x


class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""
    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, rho: float=1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma: float, sigma_next: float) ->Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) ->Tensor:
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) ->Tensor:
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x

    def inpaint(self, source: Tensor, mask: Tensor, fn: Callable, sigmas: Tensor, num_steps: int, num_resamples: int) ->Tensor:
        x = sigmas[0] * torch.randn_like(source)
        for i in range(num_steps - 1):
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
                if r < num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)
        return source * mask + x * ~mask


class DiffusionSampler(nn.Module):

    def __init__(self, diffusion: Diffusion, *, sampler: Sampler, sigma_schedule: Schedule, num_steps: Optional[int]=None, clamp: bool=True):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f'{sampler_class} incompatible with {diffusion_class}'
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    @torch.no_grad()
    def forward(self, noise: Tensor, num_steps: Optional[int]=None, **kwargs) ->Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)
        assert exists(num_steps), 'Parameter `num_steps` must be provided'
        sigmas = self.sigma_schedule(num_steps, device)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x
        return x


class DiffusionInpainter(nn.Module):

    def __init__(self, diffusion: Diffusion, *, num_steps: int, num_resamples: int, sampler: Sampler, sigma_schedule: Schedule):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor) ->Tensor:
        x = self.inpaint_fn(source=inpaint, mask=inpaint_mask, fn=self.denoise_fn, sigmas=self.sigma_schedule(self.num_steps, inpaint.device), num_steps=self.num_steps, num_resamples=self.num_resamples)
        return x


def sequential_mask(like: Tensor, start: int) ->Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):

    def __init__(self, inpainter: DiffusionInpainter, *, num_spans: int):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool=False) ->Tensor:
        half_length = start.shape[2] // 2
        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)
        for i in range(self.num_spans):
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            spans.append(second_half)
        return torch.cat(spans, dim=2)


class XDiffusion(nn.Module):

    def __init__(self, type: str, net: nn.Module, **kwargs):
        super().__init__()
        diffusion_classes = [VDiffusion, KDiffusion, VKDiffusion]
        aliases = [t.alias for t in diffusion_classes]
        message = f"type='{type}' must be one of {*aliases,}"
        assert type in aliases, message
        self.net = net
        for XDiffusion in diffusion_classes:
            if XDiffusion.alias == type:
                self.diffusion = XDiffusion(net=net, **kwargs)

    def forward(self, *args, **kwargs) ->Tensor:
        return self.diffusion(*args, **kwargs)

    def sample(self, noise: Tensor, num_steps: int, sigma_schedule: Schedule, sampler: Sampler, clamp: bool, **kwargs) ->Tensor:
        diffusion_sampler = DiffusionSampler(diffusion=self.diffusion, sampler=sampler, sigma_schedule=sigma_schedule, num_steps=num_steps, clamp=clamp)
        return diffusion_sampler(noise, **kwargs)


def Conv1d(*args, **kwargs) ->nn.Module:
    return nn.Conv1d(*args, **kwargs)


class ConvBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int=3, stride: int=1, padding: int=1, dilation: int=1, num_groups: int=8, use_norm: bool=True) ->None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) if use_norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]]=None) ->Tensor:
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x)


class MappingToScaleShift(nn.Module):

    def __init__(self, features: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(in_features=features, out_features=channels * 2))

    def forward(self, mapping: Tensor) ->Tuple[Tensor, Tensor]:
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, 'b c -> b c 1')
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift


class ResnetBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int=3, stride: int=1, padding: int=1, dilation: int=1, use_norm: bool=True, num_groups: int=8, context_mapping_features: Optional[int]=None) ->None:
        super().__init__()
        self.use_mapping = exists(context_mapping_features)
        self.block1 = ConvBlock1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, use_norm=use_norm, num_groups=num_groups)
        if self.use_mapping:
            assert exists(context_mapping_features)
            self.to_scale_shift = MappingToScaleShift(features=context_mapping_features, channels=out_channels)
        self.block2 = ConvBlock1d(in_channels=out_channels, out_channels=out_channels, use_norm=use_norm, num_groups=num_groups)
        self.to_out = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, mapping: Optional[Tensor]=None) ->Tensor:
        assert_message = 'context mapping required if context_mapping_features > 0'
        assert not self.use_mapping ^ exists(mapping), assert_message
        h = self.block1(x)
        scale_shift = None
        if self.use_mapping:
            scale_shift = self.to_scale_shift(mapping)
        h = self.block2(h, scale_shift=scale_shift)
        return h + self.to_out(x)


class RelativePositionBias(nn.Module):

    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int):
        num_buckets //= 2
        ret = (relative_position >= 0) * num_buckets
        n = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) ->Tensor:
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        relative_position_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        bias = self.relative_attention_bias(relative_position_bucket)
        bias = rearrange(bias, 'm n h -> 1 h m n')
        return bias


class AttentionBase(nn.Module):

    def __init__(self, features: int, *, head_features: int, num_heads: int, use_rel_pos: bool, rel_pos_num_buckets: Optional[int]=None, rel_pos_max_distance: Optional[int]=None):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads
        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(num_buckets=rel_pos_num_buckets, max_distance=rel_pos_max_distance, num_heads=num_heads)
        self.to_out = nn.Linear(in_features=mid_features, out_features=features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) ->Tensor:
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        sim = einsum('... n d, ... m d -> ... n m', q, k)
        sim = sim + self.rel_pos(*sim.shape[-2:]) if self.use_rel_pos else sim
        sim = sim * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('... n m, ... m d -> ... n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, features: int, *, head_features: int, num_heads: int, context_features: Optional[int]=None, use_rel_pos: bool, rel_pos_num_buckets: Optional[int]=None, rel_pos_max_distance: Optional[int]=None):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)
        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(in_features=features, out_features=mid_features, bias=False)
        self.to_kv = nn.Linear(in_features=context_features, out_features=mid_features * 2, bias=False)
        self.attention = AttentionBase(features, num_heads=num_heads, head_features=head_features, use_rel_pos=use_rel_pos, rel_pos_num_buckets=rel_pos_num_buckets, rel_pos_max_distance=rel_pos_max_distance)

    def forward(self, x: Tensor, *, context: Optional[Tensor]=None) ->Tensor:
        assert_message = 'You must provide a context when using context_features'
        assert not self.context_features or exists(context), assert_message
        context = default(context, x)
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1)
        return self.attention(q, k, v)


def FeedForward(features: int, multiplier: int) ->nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(nn.Linear(in_features=features, out_features=mid_features), nn.GELU(), nn.Linear(in_features=mid_features, out_features=features))


class TransformerBlock(nn.Module):

    def __init__(self, features: int, num_heads: int, head_features: int, multiplier: int, use_rel_pos: bool, rel_pos_num_buckets: Optional[int]=None, rel_pos_max_distance: Optional[int]=None, context_features: Optional[int]=None):
        super().__init__()
        self.use_cross_attention = exists(context_features) and context_features > 0
        self.attention = Attention(features=features, num_heads=num_heads, head_features=head_features, use_rel_pos=use_rel_pos, rel_pos_num_buckets=rel_pos_num_buckets, rel_pos_max_distance=rel_pos_max_distance)
        if self.use_cross_attention:
            self.cross_attention = Attention(features=features, num_heads=num_heads, head_features=head_features, context_features=context_features, use_rel_pos=use_rel_pos, rel_pos_num_buckets=rel_pos_num_buckets, rel_pos_max_distance=rel_pos_max_distance)
        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor]=None) ->Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        x = self.feed_forward(x) + x
        return x


class Transformer1d(nn.Module):

    def __init__(self, num_layers: int, channels: int, num_heads: int, head_features: int, multiplier: int, use_rel_pos: bool=False, rel_pos_num_buckets: Optional[int]=None, rel_pos_max_distance: Optional[int]=None, context_features: Optional[int]=None):
        super().__init__()
        self.to_in = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-06, affine=True), Conv1d(in_channels=channels, out_channels=channels, kernel_size=1), Rearrange('b c t -> b t c'))
        self.blocks = nn.ModuleList([TransformerBlock(features=channels, head_features=head_features, num_heads=num_heads, multiplier=multiplier, context_features=context_features, use_rel_pos=use_rel_pos, rel_pos_num_buckets=rel_pos_num_buckets, rel_pos_max_distance=rel_pos_max_distance) for i in range(num_layers)])
        self.to_out = nn.Sequential(Rearrange('b t c -> b c t'), Conv1d(in_channels=channels, out_channels=channels, kernel_size=1))

    def forward(self, x: Tensor, *, context: Optional[Tensor]=None) ->Tensor:
        x = self.to_in(x)
        for block in self.blocks:
            x = block(x, context=context)
        x = self.to_out(x)
        return x


class BottleneckBlock1d(nn.Module):

    def __init__(self, channels: int, *, num_groups: int, num_transformer_blocks: int=0, attention_heads: Optional[int]=None, attention_features: Optional[int]=None, attention_multiplier: Optional[int]=None, attention_use_rel_pos: Optional[bool]=None, attention_rel_pos_max_distance: Optional[int]=None, attention_rel_pos_num_buckets: Optional[int]=None, context_mapping_features: Optional[int]=None, context_embedding_features: Optional[int]=None):
        super().__init__()
        self.use_transformer = num_transformer_blocks > 0
        self.pre_block = ResnetBlock1d(in_channels=channels, out_channels=channels, num_groups=num_groups, context_mapping_features=context_mapping_features)
        if self.use_transformer:
            assert exists(attention_heads) and exists(attention_features) and exists(attention_multiplier) and exists(attention_use_rel_pos)
            self.transformer = Transformer1d(num_layers=num_transformer_blocks, channels=channels, num_heads=attention_heads, head_features=attention_features, multiplier=attention_multiplier, context_features=context_embedding_features, use_rel_pos=attention_use_rel_pos, rel_pos_num_buckets=attention_rel_pos_num_buckets, rel_pos_max_distance=attention_rel_pos_max_distance)
        self.post_block = ResnetBlock1d(in_channels=channels, out_channels=channels, num_groups=num_groups, context_mapping_features=context_mapping_features)

    def forward(self, x: Tensor, *, mapping: Optional[Tensor]=None, embedding: Optional[Tensor]=None) ->Tensor:
        x = self.pre_block(x, mapping=mapping)
        if self.use_transformer:
            x = self.transformer(x, context=embedding)
        x = self.post_block(x, mapping=mapping)
        return x


def Downsample1d(in_channels: int, out_channels: int, factor: int, kernel_multiplier: int=2) ->nn.Module:
    assert kernel_multiplier % 2 == 0, 'Kernel multiplier must be even'
    return Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=factor * kernel_multiplier + 1, stride=factor, padding=factor * (kernel_multiplier // 2))


class DownsampleBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, factor: int, num_groups: int, num_layers: int, kernel_multiplier: int=2, use_pre_downsample: bool=True, use_skip: bool=False, extract_channels: int=0, context_channels: int=0, num_transformer_blocks: int=0, attention_heads: Optional[int]=None, attention_features: Optional[int]=None, attention_multiplier: Optional[int]=None, attention_use_rel_pos: Optional[bool]=None, attention_rel_pos_max_distance: Optional[int]=None, attention_rel_pos_num_buckets: Optional[int]=None, context_mapping_features: Optional[int]=None, context_embedding_features: Optional[int]=None):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_transformer = num_transformer_blocks > 0
        self.use_extract = extract_channels > 0
        self.use_context = context_channels > 0
        channels = out_channels if use_pre_downsample else in_channels
        self.downsample = Downsample1d(in_channels=in_channels, out_channels=out_channels, factor=factor, kernel_multiplier=kernel_multiplier)
        self.blocks = nn.ModuleList([ResnetBlock1d(in_channels=channels + context_channels if i == 0 else channels, out_channels=channels, num_groups=num_groups, context_mapping_features=context_mapping_features) for i in range(num_layers)])
        if self.use_transformer:
            assert exists(attention_heads) and exists(attention_features) and exists(attention_multiplier) and exists(attention_use_rel_pos)
            self.transformer = Transformer1d(num_layers=num_transformer_blocks, channels=channels, num_heads=attention_heads, head_features=attention_features, multiplier=attention_multiplier, context_features=context_embedding_features, use_rel_pos=attention_use_rel_pos, rel_pos_num_buckets=attention_rel_pos_num_buckets, rel_pos_max_distance=attention_rel_pos_max_distance)
        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(in_channels=out_channels, out_channels=extract_channels, num_groups=num_extract_groups)

    def forward(self, x: Tensor, *, mapping: Optional[Tensor]=None, channels: Optional[Tensor]=None, embedding: Optional[Tensor]=None) ->Union[Tuple[Tensor, List[Tensor]], Tensor]:
        if self.use_pre_downsample:
            x = self.downsample(x)
        if self.use_context and exists(channels):
            x = torch.cat([x, channels], dim=1)
        skips = []
        for block in self.blocks:
            x = block(x, mapping=mapping)
            skips += [x] if self.use_skip else []
        if self.use_transformer:
            x = self.transformer(x, context=embedding)
            skips += [x] if self.use_skip else []
        if not self.use_pre_downsample:
            x = self.downsample(x)
        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted
        return (x, skips) if self.use_skip else x


class Patcher(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, context_mapping_features: Optional[int]=None):
        super().__init__()
        assert_message = f'out_channels must be divisible by patch_size ({patch_size})'
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size
        self.block = ResnetBlock1d(in_channels=in_channels, out_channels=out_channels // patch_size, num_groups=1, context_mapping_features=context_mapping_features)

    def forward(self, x: Tensor, mapping: Optional[Tensor]=None) ->Tensor:
        x = self.block(x, mapping)
        x = rearrange(x, 'b c (l p) -> b (c p) l', p=self.patch_size)
        return x


def closest_power_2(x: float) ->int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(self, num_fft: int=1023, hop_length: int=256, window_length: Optional[int]=None, length: Optional[int]=None, use_complex: bool=False):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer('window', torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) ->Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, 'b c t -> (b c) t')
        stft = torch.stft(wave, n_fft=self.num_fft, hop_length=self.hop_length, win_length=self.window_length, window=self.window, return_complex=True, normalized=True)
        if self.use_complex:
            stft_a, stft_b = stft.real, stft.imag
        else:
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase
        return rearrange(stft_a, '(b c) f l -> b c f l', b=b), rearrange(stft_b, '(b c) f l -> b c f l', b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) ->Tensor:
        b, l = stft_a.shape[0], stft_a.shape[-1]
        length = closest_power_2(l * self.hop_length)
        stft_a = rearrange(stft_a, 'b c f l -> (b c) f l')
        stft_b = rearrange(stft_b, 'b c f l -> (b c) f l')
        if self.use_complex:
            real, imag = stft_a, stft_b
        else:
            magnitude, phase = stft_a, stft_b
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        stft = torch.stack([real, imag], dim=-1)
        wave = torch.istft(stft, n_fft=self.num_fft, hop_length=self.hop_length, win_length=self.window_length, window=self.window, length=default(self.length, length), normalized=True)
        return rearrange(wave, '(b c) t -> b c t', b=b)

    def encode1d(self, wave: Tensor, stacked: bool=True) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        stft_a, stft_b = self.encode(wave)
        stft_a = rearrange(stft_a, 'b c f l -> b (c f) l')
        stft_b = rearrange(stft_b, 'b c f l -> b (c f) l')
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) ->Tensor:
        f = self.num_fft // 2 + 1
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        stft_a = rearrange(stft_a, 'b (c f) l -> b c f l', f=f)
        stft_b = rearrange(stft_b, 'b (c f) l -> b c f l', f=f)
        return self.decode(stft_a, stft_b)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) ->Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) ->nn.Module:
    return nn.Sequential(LearnedPositionalEmbedding(dim), nn.Linear(in_features=dim + 1, out_features=out_features))


class Unpatcher(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, context_mapping_features: Optional[int]=None):
        super().__init__()
        assert_message = f'in_channels must be divisible by patch_size ({patch_size})'
        assert in_channels % patch_size == 0, assert_message
        self.patch_size = patch_size
        self.block = ResnetBlock1d(in_channels=in_channels // patch_size, out_channels=out_channels, num_groups=1, context_mapping_features=context_mapping_features)

    def forward(self, x: Tensor, mapping: Optional[Tensor]=None) ->Tensor:
        x = rearrange(x, ' b (c p) l -> b c (l p) ', p=self.patch_size)
        x = self.block(x, mapping)
        return x


def ConvTranspose1d(*args, **kwargs) ->nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)


def Upsample1d(in_channels: int, out_channels: int, factor: int, use_nearest: bool=False) ->nn.Module:
    if factor == 1:
        return Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    if use_nearest:
        return nn.Sequential(nn.Upsample(scale_factor=factor, mode='nearest'), Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
    else:
        return ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=factor * 2, stride=factor, padding=factor // 2 + factor % 2, output_padding=factor % 2)


class UpsampleBlock1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, factor: int, num_layers: int, num_groups: int, use_nearest: bool=False, use_pre_upsample: bool=False, use_skip: bool=False, skip_channels: int=0, use_skip_scale: bool=False, extract_channels: int=0, num_transformer_blocks: int=0, attention_heads: Optional[int]=None, attention_features: Optional[int]=None, attention_multiplier: Optional[int]=None, attention_use_rel_pos: Optional[bool]=None, attention_rel_pos_max_distance: Optional[int]=None, attention_rel_pos_num_buckets: Optional[int]=None, context_mapping_features: Optional[int]=None, context_embedding_features: Optional[int]=None):
        super().__init__()
        self.use_extract = extract_channels > 0
        self.use_pre_upsample = use_pre_upsample
        self.use_transformer = num_transformer_blocks > 0
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0
        channels = out_channels if use_pre_upsample else in_channels
        self.blocks = nn.ModuleList([ResnetBlock1d(in_channels=channels + skip_channels, out_channels=channels, num_groups=num_groups, context_mapping_features=context_mapping_features) for _ in range(num_layers)])
        if self.use_transformer:
            assert exists(attention_heads) and exists(attention_features) and exists(attention_multiplier) and exists(attention_use_rel_pos)
            self.transformer = Transformer1d(num_layers=num_transformer_blocks, channels=channels, num_heads=attention_heads, head_features=attention_features, multiplier=attention_multiplier, context_features=context_embedding_features, use_rel_pos=attention_use_rel_pos, rel_pos_num_buckets=attention_rel_pos_num_buckets, rel_pos_max_distance=attention_rel_pos_max_distance)
        self.upsample = Upsample1d(in_channels=in_channels, out_channels=out_channels, factor=factor, use_nearest=use_nearest)
        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(in_channels=out_channels, out_channels=extract_channels, num_groups=num_extract_groups)

    def add_skip(self, x: Tensor, skip: Tensor) ->Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(self, x: Tensor, *, skips: Optional[List[Tensor]]=None, mapping: Optional[Tensor]=None, embedding: Optional[Tensor]=None) ->Union[Tuple[Tensor, Tensor], Tensor]:
        if self.use_pre_upsample:
            x = self.upsample(x)
        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, mapping=mapping)
        if self.use_transformer:
            x = self.transformer(x, context=embedding)
        if not self.use_pre_upsample:
            x = self.upsample(x)
        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted
        return x


def group_dict_by_prefix(prefix: str, d: Dict) ->Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool=False) ->Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix):]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


class UNet1d(nn.Module):

    def __init__(self, in_channels: int, channels: int, multipliers: Sequence[int], factors: Sequence[int], num_blocks: Sequence[int], attentions: Sequence[int], patch_size: int=1, resnet_groups: int=8, use_context_time: bool=True, kernel_multiplier_downsample: int=2, use_nearest_upsample: bool=False, use_skip_scale: bool=True, use_stft: bool=False, use_stft_context: bool=False, out_channels: Optional[int]=None, context_features: Optional[int]=None, context_features_multiplier: int=4, context_channels: Optional[Sequence[int]]=None, context_embedding_features: Optional[int]=None, **kwargs):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        context_channels = list(default(context_channels, []))
        num_layers = len(multipliers) - 1
        use_context_features = exists(context_features)
        use_context_channels = len(context_channels) > 0
        context_mapping_features = None
        attention_kwargs, kwargs = groupby('attention_', kwargs, keep_prefix=True)
        self.num_layers = num_layers
        self.use_context_time = use_context_time
        self.use_context_features = use_context_features
        self.use_context_channels = use_context_channels
        self.use_stft = use_stft
        self.use_stft_context = use_stft_context
        self.context_features = context_features
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        context_channels = context_channels + [0] * context_channels_pad_length
        self.context_channels = context_channels
        self.context_embedding_features = context_embedding_features
        if use_context_channels:
            has_context = [(c > 0) for c in context_channels]
            self.has_context = has_context
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]
        assert len(factors) == num_layers and len(attentions) >= num_layers and len(num_blocks) == num_layers
        if use_context_time or use_context_features:
            context_mapping_features = channels * context_features_multiplier
            self.to_mapping = nn.Sequential(nn.Linear(context_mapping_features, context_mapping_features), nn.GELU(), nn.Linear(context_mapping_features, context_mapping_features), nn.GELU())
        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(TimePositionalEmbedding(dim=channels, out_features=context_mapping_features), nn.GELU())
        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(nn.Linear(in_features=context_features, out_features=context_mapping_features), nn.GELU())
        if use_stft:
            stft_kwargs, kwargs = groupby('stft_', kwargs)
            assert 'num_fft' in stft_kwargs, 'stft_num_fft required if use_stft=True'
            stft_channels = (stft_kwargs['num_fft'] // 2 + 1) * 2
            in_channels *= stft_channels
            out_channels *= stft_channels
            context_channels[0] *= stft_channels if use_stft_context else 1
            assert exists(in_channels) and exists(out_channels)
            self.stft = STFT(**stft_kwargs)
        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"
        self.to_in = Patcher(in_channels=in_channels + context_channels[0], out_channels=channels * multipliers[0], patch_size=patch_size, context_mapping_features=context_mapping_features)
        self.downsamples = nn.ModuleList([DownsampleBlock1d(in_channels=channels * multipliers[i], out_channels=channels * multipliers[i + 1], context_mapping_features=context_mapping_features, context_channels=context_channels[i + 1], context_embedding_features=context_embedding_features, num_layers=num_blocks[i], factor=factors[i], kernel_multiplier=kernel_multiplier_downsample, num_groups=resnet_groups, use_pre_downsample=True, use_skip=True, num_transformer_blocks=attentions[i], **attention_kwargs) for i in range(num_layers)])
        self.bottleneck = BottleneckBlock1d(channels=channels * multipliers[-1], context_mapping_features=context_mapping_features, context_embedding_features=context_embedding_features, num_groups=resnet_groups, num_transformer_blocks=attentions[-1], **attention_kwargs)
        self.upsamples = nn.ModuleList([UpsampleBlock1d(in_channels=channels * multipliers[i + 1], out_channels=channels * multipliers[i], context_mapping_features=context_mapping_features, context_embedding_features=context_embedding_features, num_layers=num_blocks[i] + (1 if attentions[i] else 0), factor=factors[i], use_nearest=use_nearest_upsample, num_groups=resnet_groups, use_skip_scale=use_skip_scale, use_pre_upsample=False, use_skip=True, skip_channels=channels * multipliers[i + 1], num_transformer_blocks=attentions[i], **attention_kwargs) for i in reversed(range(num_layers))])
        self.to_out = Unpatcher(in_channels=channels * multipliers[0], out_channels=out_channels, patch_size=patch_size, context_mapping_features=context_mapping_features)

    def get_channels(self, channels_list: Optional[Sequence[Tensor]]=None, layer: int=0) ->Optional[Tensor]:
        """Gets context channels at `layer` and checks that shape is correct"""
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), 'Missing context'
        channels_id = self.channels_ids[layer]
        channels = channels_list[channels_id]
        message = f'Missing context for layer {layer} at index {channels_id}'
        assert exists(channels), message
        num_channels = self.context_channels[layer]
        message = f'Expected context with {num_channels} channels at idx {channels_id}'
        assert channels.shape[1] == num_channels, message
        channels = self.stft.encode1d(channels) if self.use_stft_context else channels
        return channels

    def get_mapping(self, time: Optional[Tensor]=None, features: Optional[Tensor]=None) ->Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        if self.use_context_time:
            assert_message = 'use_context_time=True but no time features provided'
            assert exists(time), assert_message
            items += [self.to_time(time)]
        if self.use_context_features:
            assert_message = 'context_features exists but no features provided'
            assert exists(features), assert_message
            items += [self.to_features(features)]
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), 'n b m -> b m', 'sum')
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(self, x: Tensor, time: Optional[Tensor]=None, *, features: Optional[Tensor]=None, channels_list: Optional[Sequence[Tensor]]=None, embedding: Optional[Tensor]=None) ->Tensor:
        channels = self.get_channels(channels_list, layer=0)
        x = self.stft.encode1d(x) if self.use_stft else x
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping)
        skips_list = [x]
        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i + 1)
            x, skips = downsample(x, mapping=mapping, channels=channels, embedding=embedding)
            skips_list += [skips]
        x = self.bottleneck(x, mapping=mapping, embedding=embedding)
        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding)
        x += skips_list.pop()
        x = self.to_out(x, mapping)
        x = self.stft.decode1d(x) if self.use_stft else x
        return x


class FixedEmbedding(nn.Module):

    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) ->Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = 'Input sequence length must be <= max_length'
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, 'n d -> b n d', b=batch_size)
        return fixed_embedding


def rand_bool(shape: Any, proba: float, device: Any=None) ->Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device))


class UNetCFG1d(UNet1d):
    """UNet1d with Classifier-Free Guidance"""

    def __init__(self, context_embedding_max_length: int, context_embedding_features: int, **kwargs):
        super().__init__(context_embedding_features=context_embedding_features, **kwargs)
        self.fixed_embedding = FixedEmbedding(max_length=context_embedding_max_length, features=context_embedding_features)

    def forward(self, x: Tensor, time: Tensor, *, embedding: Tensor, embedding_scale: float=1.0, embedding_mask_proba: float=0.0, **kwargs) ->Tensor:
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            batch_mask = rand_bool(shape=(b, 1, 1), proba=embedding_mask_proba, device=device)
            embedding = torch.where(batch_mask, fixed_embedding, embedding)
        if embedding_scale != 1.0:
            out = super().forward(x, time, embedding=embedding, **kwargs)
            out_masked = super().forward(x, time, embedding=fixed_embedding, **kwargs)
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return super().forward(x, time, embedding=embedding, **kwargs)


class NumberEmbedder(nn.Module):

    def __init__(self, features: int, dim: int=256):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) ->Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, '... -> (...)')
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x


class UNetNCCA1d(UNet1d):
    """UNet1d with Noise Channel Conditioning Augmentation"""

    def __init__(self, context_features: int, **kwargs):
        super().__init__(context_features=context_features, **kwargs)
        self.embedder = NumberEmbedder(features=context_features)

    def expand(self, x: Any, shape: Tuple[int, ...]) ->Tensor:
        x = x if torch.is_tensor(x) else torch.tensor(x)
        return x.expand(shape)

    def forward(self, x: Tensor, time: Tensor, *, channels_list: Sequence[Tensor], channels_augmentation: Union[bool, Sequence[bool], Sequence[Sequence[bool]], Tensor]=False, channels_scale: Union[float, Sequence[float], Sequence[Sequence[float]], Tensor]=0, **kwargs) ->Tensor:
        b, n = x.shape[0], len(channels_list)
        channels_augmentation = self.expand(channels_augmentation, shape=(b, n))
        channels_scale = self.expand(channels_scale, shape=(b, n))
        for i in range(n):
            scale = channels_scale[:, i] * channels_augmentation[:, i]
            scale = rearrange(scale, 'b -> b 1 1')
            item = channels_list[i]
            channels_list[i] = torch.randn_like(item) * scale + item * (1 - scale)
        channels_scale_emb = self.embedder(channels_scale)
        channels_scale_emb = reduce(channels_scale_emb, 'b n d -> b d', 'sum')
        return super().forward(x=x, time=time, channels_list=channels_list, features=channels_scale_emb, **kwargs)


class UNetAll1d(UNetCFG1d, UNetNCCA1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return UNetCFG1d.forward(self, *args, **kwargs)


def XUNet1d(type: str='base', **kwargs) ->UNet1d:
    if type == 'base':
        return UNet1d(**kwargs)
    elif type == 'all':
        return UNetAll1d(**kwargs)
    elif type == 'cfg':
        return UNetCFG1d(**kwargs)
    elif type == 'ncca':
        return UNetNCCA1d(**kwargs)
    else:
        raise ValueError(f'Unknown XUNet1d type: {type}')


class Model1d(nn.Module):

    def __init__(self, unet_type: str='base', **kwargs):
        super().__init__()
        diffusion_kwargs, kwargs = groupby('diffusion_', kwargs)
        self.unet = XUNet1d(type=unet_type, **kwargs)
        self.diffusion = XDiffusion(net=self.unet, **diffusion_kwargs)

    def forward(self, x: Tensor, **kwargs) ->Tensor:
        return self.diffusion(x, **kwargs)

    def sample(self, *args, **kwargs) ->Tensor:
        return self.diffusion.sample(*args, **kwargs)


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) ->Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def resample(waveforms: Tensor, factor_in: int, factor_out: int, rolloff: float=0.99, lowpass_filter_width: int=6) ->Tensor:
    """Resamples a waveform using sinc interpolation, adapted from torchaudio"""
    b, _, length = waveforms.shape
    length_target = int(factor_out * length / factor_in)
    d = dict(device=waveforms.device, dtype=waveforms.dtype)
    base_factor = min(factor_in, factor_out) * rolloff
    width = ceil(lowpass_filter_width * factor_in / base_factor)
    idx = torch.arange(-width, width + factor_in, **d)[None, None] / factor_in
    t = torch.arange(0, -factor_out, step=-1, **d)[:, None, None] / factor_out + idx
    t = (t * base_factor).clamp(-lowpass_filter_width, lowpass_filter_width) * pi
    window = torch.cos(t / lowpass_filter_width / 2) ** 2
    scale = base_factor / factor_in
    kernels = torch.where(t == 0, torch.tensor(1.0), t.sin() / t)
    kernels *= window * scale
    waveforms = rearrange(waveforms, 'b c t -> (b c) t')
    waveforms = F.pad(waveforms, (width, width + factor_in))
    resampled = F.conv1d(waveforms[:, None], kernels, stride=factor_in)
    resampled = rearrange(resampled, '(b c) k l -> b c (l k)', b=b)
    return resampled[..., :length_target]


def downsample(waveforms: Tensor, factor: int, **kwargs) ->Tensor:
    return resample(waveforms, factor_in=factor, factor_out=1, **kwargs)


def to_list(val: Union[T, Sequence[T]]) ->List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]


def upsample(waveforms: Tensor, factor: int, **kwargs) ->Tensor:
    return resample(waveforms, factor_in=1, factor_out=factor, **kwargs)


class DiffusionUpsampler1d(Model1d):

    def __init__(self, in_channels: int, factor: Union[int, Sequence[int]], factor_features: Optional[int]=None, *args, **kwargs):
        self.factors = to_list(factor)
        self.use_conditioning = exists(factor_features)
        default_kwargs = dict(in_channels=in_channels, context_channels=[in_channels], context_features=factor_features if self.use_conditioning else None)
        super().__init__(*args, **{**default_kwargs, **kwargs})
        if self.use_conditioning:
            assert exists(factor_features)
            self.to_features = SinusoidalEmbedding(dim=factor_features)

    def random_reupsample(self, x: Tensor) ->Tuple[Tensor, Tensor]:
        batch_size, device, factors = x.shape[0], x.device, self.factors
        random_factors = torch.randint(0, len(factors), (batch_size,), device=device)
        x = x.clone()
        for i, factor in enumerate(factors):
            n = torch.count_nonzero(random_factors == i)
            if n > 0:
                waveforms = x[random_factors == i]
                downsampled = downsample(waveforms, factor=factor)
                reupsampled = upsample(downsampled, factor=factor)
                x[random_factors == i] = reupsampled
        return x, random_factors

    def forward(self, x: Tensor, **kwargs) ->Tensor:
        channels, factors = self.random_reupsample(x)
        features = self.to_features(factors) if self.use_conditioning else None
        return self.diffusion(x, channels_list=[channels], features=features, **kwargs)

    def sample(self, undersampled: Tensor, factor: Optional[int]=None, *args, **kwargs):
        batch_size, device = undersampled.shape[0], undersampled.device
        factor = default(factor, self.factors[0])
        channels = upsample(undersampled, factor=factor)
        factors = torch.tensor([factor] * batch_size, device=device)
        features = self.to_features(factors) if self.use_conditioning else None
        noise = torch.randn_like(channels)
        default_kwargs = dict(channels_list=[channels], features=features)
        return super().sample(noise, **{**default_kwargs, **kwargs})


class DiffusionVocoder1d(Model1d):

    def __init__(self, in_channels: int, stft_num_fft: int, **kwargs):
        self.frequency_channels = stft_num_fft // 2 + 1
        spectrogram_channels = in_channels * self.frequency_channels
        stft_kwargs, kwargs = groupby('stft_', kwargs)
        default_kwargs = dict(in_channels=spectrogram_channels, context_channels=[spectrogram_channels])
        super().__init__(**{**default_kwargs, **kwargs})
        self.stft = STFT(num_fft=stft_num_fft, **stft_kwargs)

    def forward_wave(self, x: Tensor, **kwargs) ->Tensor:
        magnitude, phase = self.stft.encode(x)
        return self(magnitude, phase, **kwargs)

    def forward(self, magnitude: Tensor, phase: Tensor, **kwargs) ->Tensor:
        magnitude = rearrange(magnitude, 'b c f t -> b (c f) t')
        phase = rearrange(phase, 'b c f t -> b (c f) t')
        return self.diffusion(phase / pi, channels_list=[magnitude], **kwargs)

    def sample(self, magnitude: Tensor, **kwargs):
        b, c, f, t, device = *magnitude.shape, magnitude.device
        magnitude_flat = rearrange(magnitude, 'b c f t -> b (c f) t')
        noise = torch.randn((b, c * f, t), device=device)
        default_kwargs = dict(channels_list=[magnitude_flat])
        phase_flat = super().sample(noise, **{**default_kwargs, **kwargs})
        phase = rearrange(phase_flat, 'b (c f) t -> b c f t', c=c)
        wave = self.stft.decode(magnitude, phase * pi)
        return wave


class DiffusionUpphaser1d(DiffusionUpsampler1d):

    def __init__(self, **kwargs):
        stft_kwargs, kwargs = groupby('stft_', kwargs)
        super().__init__(**kwargs)
        self.stft = STFT(**stft_kwargs)

    def random_rephase(self, x: Tensor) ->Tensor:
        magnitude, phase = self.stft.encode(x)
        phase_random = (torch.rand_like(phase) - 0.5) * 2 * pi
        wave = self.stft.decode(magnitude, phase_random)
        return wave

    def forward(self, x: Tensor, **kwargs) ->Tensor:
        rephased = self.random_rephase(x)
        resampled, factors = self.random_reupsample(rephased)
        features = self.to_features(factors) if self.use_conditioning else None
        return self.diffusion(x, channels_list=[resampled], features=features, **kwargs)


class DiffusionAR1d(Model1d):

    def __init__(self, in_channels: int, chunk_length: int, upsample: int=0, dropout: float=0.05, verbose: int=0, **kwargs):
        self.in_channels = in_channels
        self.chunk_length = chunk_length
        self.dropout = dropout
        self.upsample = upsample
        self.verbose = verbose
        super().__init__(in_channels=in_channels, context_channels=[in_channels * (2 if upsample > 0 else 1)], **kwargs)

    def reupsample(self, x: Tensor) ->Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample)
        x = upsample(x, factor=self.upsample)
        return x

    def forward(self, x: Tensor, **kwargs) ->Tensor:
        b, _, t, device = *x.shape, x.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert num_chunks >= 2, 'Input tensor length must be >= chunk_length * 2'
        chunk_index = randint(0, num_chunks - 2)
        chunk_pos = cl * (chunk_index + 1)
        chunk_prev = x[:, :, cl * chunk_index:chunk_pos]
        chunk_curr = x[:, :, chunk_pos:cl * (chunk_index + 2)]
        if self.dropout > 0:
            batch_mask = rand_bool(shape=(b, 1, 1), proba=self.dropout, device=device)
            chunk_zeros = torch.zeros_like(chunk_prev)
            chunk_prev = torch.where(batch_mask, chunk_zeros, chunk_prev)
        if self.upsample > 0:
            chunk_reupsampled = self.reupsample(chunk_curr)
            channels_list = [torch.cat([chunk_prev, chunk_reupsampled], dim=1)]
        else:
            channels_list = [chunk_prev]
        return self.diffusion(chunk_curr, channels_list=channels_list, **kwargs)

    def sample(self, x: Tensor, start: Optional[Tensor]=None, **kwargs) ->Tensor:
        noise = x
        if self.upsample > 0:
            upsampled = upsample(x, factor=self.upsample)
            noise = torch.randn_like(upsampled)
        b, c, t, device = *noise.shape, noise.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert c == self.in_channels
        assert t % cl == 0, 'noise must be divisible by chunk_length'
        if exists(start):
            chunk_prev = start[:, :, -cl:]
        else:
            chunk_prev = torch.zeros(b, c, cl)
        chunks = []
        for i in tqdm(range(num_chunks), disable=self.verbose == 0):
            chunk_start, chunk_end = cl * i, cl * (i + 1)
            noise_curr = noise[:, :, chunk_start:chunk_end]
            if self.upsample > 0:
                chunk_upsampled = upsampled[:, :, chunk_start:chunk_end]
                channels_list = [torch.cat([chunk_prev, chunk_upsampled], dim=1)]
            else:
                channels_list = [chunk_prev]
            default_kwargs = dict(channels_list=channels_list)
            chunk_curr = super().sample(noise_curr, **{**default_kwargs, **kwargs})
            chunks += [chunk_curr]
            chunk_prev = chunk_curr
        return rearrange(chunks, 'l b c t -> b c (l t)')


class UniformDistribution(Distribution):

    def __call__(self, num_samples: int, device: torch.device=torch.device('cpu')):
        return torch.rand(num_samples, device=device)


def get_default_model_kwargs():
    return dict(channels=128, patch_size=16, multipliers=[1, 2, 4, 4, 4, 4, 4], factors=[4, 4, 4, 2, 2, 2], num_blocks=[2, 2, 2, 2, 2, 2], attentions=[0, 0, 0, 1, 1, 1, 1], attention_heads=8, attention_features=64, attention_multiplier=2, attention_use_rel_pos=False, diffusion_type='v', diffusion_sigma_distribution=UniformDistribution())


def get_default_sampling_kwargs():
    return dict(sigma_schedule=LinearSchedule(), sampler=VSampler(), clamp=True)


class AudioDiffusionModel(Model1d):

    def __init__(self, **kwargs):
        super().__init__(**{**get_default_model_kwargs(), **kwargs})

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionUpsampler(DiffusionUpsampler1d):

    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(**get_default_model_kwargs(), in_channels=in_channels, context_channels=[in_channels])
        super().__init__(**{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionConditional(Model1d):

    def __init__(self, embedding_features: int, embedding_max_length: int, embedding_mask_proba: float=0.1, **kwargs):
        self.embedding_mask_proba = embedding_mask_proba
        default_kwargs = dict(**get_default_model_kwargs(), unet_type='cfg', context_embedding_features=embedding_features, context_embedding_max_length=embedding_max_length)
        super().__init__(**{**default_kwargs, **kwargs})

    def forward(self, *args, **kwargs):
        default_kwargs = dict(embedding_mask_proba=self.embedding_mask_proba)
        return super().forward(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(**get_default_sampling_kwargs(), embedding_scale=5.0)
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionVocoder(DiffusionVocoder1d):

    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(in_channels=in_channels, stft_num_fft=1023, stft_hop_length=256, channels=512, multipliers=[3, 2, 1, 1, 1, 1, 1, 1], factors=[1, 2, 2, 2, 2, 2, 2], num_blocks=[1, 1, 1, 1, 1, 1, 1], attentions=[0, 0, 0, 0, 1, 1, 1], attention_heads=8, attention_features=64, attention_multiplier=2, attention_use_rel_pos=False, diffusion_type='v', diffusion_sigma_distribution=UniformDistribution())
        super().__init__(**{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(**get_default_sampling_kwargs())
        return super().sample(*args, **{**default_kwargs, **kwargs})


class AudioDiffusionUpphaser(DiffusionUpphaser1d):

    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(**get_default_model_kwargs(), in_channels=in_channels, context_channels=[in_channels], factor=1)
        super().__init__(**{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class ConditionedSequential(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(*modules)

    def forward(self, x: Tensor, mapping: Optional[Tensor]=None):
        for module in self.module_list:
            x = module(x, mapping)
        return x


class T5Embedder(nn.Module):

    def __init__(self, model: str='t5-base', max_length: int=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.transformer = T5EncoderModel.from_pretrained(model)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, texts: List[str]) ->Tensor:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        device = next(self.transformer.parameters()).device
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        self.transformer.eval()
        embedding = self.transformer(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return embedding


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ADPM2Sampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (AEulerSampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (ConditionedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_archinetai_audio_diffusion_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

