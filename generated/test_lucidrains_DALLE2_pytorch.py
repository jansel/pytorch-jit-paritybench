import sys
_module = sys.modules[__name__]
del sys
dalle2_pytorch = _module
cli = _module
dalle2_pytorch = _module
dataloaders = _module
decoder_loader = _module
prior_loader = _module
simple_image_only_dataloader = _module
optimizer = _module
tokenizer = _module
trackers = _module
train_configs = _module
trainer = _module
utils = _module
version = _module
vqgan_vae = _module
vqgan_vae_trainer = _module
setup = _module
train_decoder = _module
train_diffusion_prior = _module

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


import torchvision.transforms as T


from functools import reduce


import math


import random


from functools import partial


from functools import wraps


from collections import namedtuple


import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


from torch import nn


from torch import einsum


from torch.utils.data import DataLoader


import numpy as np


from math import ceil


from torch import from_numpy


from torch.utils.data import IterableDataset


from torch.utils import data


from torchvision import transforms


from torchvision import utils


from torch.optim import AdamW


from torch.optim import Adam


from functools import lru_cache


from itertools import zip_longest


from typing import Any


from typing import Optional


from typing import List


from typing import Union


from torchvision import transforms as T


from typing import Tuple


from typing import Dict


from typing import TypeVar


import time


import copy


from collections.abc import Iterable


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from math import sqrt


from torch.autograd import grad as torch_grad


import torchvision


from random import choice


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torchvision.datasets import ImageFolder


from torchvision.utils import make_grid


from torchvision.utils import save_image


def exists(val):
    return val is not None


def resize_image_to(image, target_image_size, clamp_range=None, nearest=False, **kwargs):
    orig_image_size = image.shape[-1]
    if orig_image_size == target_image_size:
        return image
    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors=scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode='nearest')
    if exists(clamp_range):
        out = out.clamp(*clamp_range)
    return out


class BaseClipAdapter(nn.Module):

    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert image_size >= self.image_size, f'you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}'
        return resize_image_to(image, self.image_size)

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError


EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])


EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])


def l2norm(t):
    return F.normalize(t, dim=-1)


class XClipAdapter(BaseClipAdapter):

    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        encoder_output = self.clip.text_transformer(text)
        encoder_output_is_cls = encoder_output.ndim == 3
        text_cls, text_encodings = (encoder_output[:, 0], encoder_output[:, 1:]) if encoder_output_is_cls else (encoder_output, None)
        text_embed = self.clip.to_text_latent(text_cls)
        if exists(text_encodings):
            text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        return EmbeddedText(l2norm(text_embed), text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
        encoder_output = self.clip.visual_transformer(image)
        image_cls, image_encodings = encoder_output[:, 0], encoder_output[:, 1:]
        image_embed = self.clip.to_visual_latent(image_cls)
        return EmbeddedImage(l2norm(image_embed), image_encodings)


class CoCaAdapter(BaseClipAdapter):

    @property
    def dim_latent(self):
        return self.clip.dim

    @property
    def image_size(self):
        assert 'image_size' in self.overrides
        return self.overrides['image_size']

    @property
    def image_channels(self):
        assert 'image_channels' in self.overrides
        return self.overrides['image_channels']

    @property
    def max_text_len(self):
        assert 'max_text_len' in self.overrides
        return self.overrides['max_text_len']

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        text_embed, text_encodings = self.clip.embed_text(text)
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        return EmbeddedText(text_embed, text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
        image_embed, image_encodings = self.clip.embed_image(image)
        return EmbeddedImage(image_embed, image_encodings)


class OpenAIClipAdapter(BaseClipAdapter):

    def __init__(self, name='ViT-B/32'):
        openai_clip, preprocess = clip.load(name)
        super().__init__(openai_clip)
        self.eos_id = 49407
        text_attention_final = self.find_layer('ln_final')
        self.dim_latent_ = text_attention_final.weight.shape[0]
        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self, layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return
        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self.dim_latent_

    @property
    def image_size(self):
        return self.clip.visual.input_resolution

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        is_eos_id = text == self.eos_id
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared
        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)


class OpenClipAdapter(BaseClipAdapter):

    def __init__(self, name='ViT-B/32', pretrained='laion400m_e32'):
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        super().__init__(clip)
        self.eos_id = 49407
        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]
        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self, layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return
        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        is_eos_id = text == self.eos_id
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared
        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


def default(val, d):
    return val if exists(val) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(nn.Module):

    def __init__(self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma=0.0, p2_loss_weight_k=1):
        super().__init__()
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == 'jsd':
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        self.loss_type = loss_type
        self.loss_fn = loss_fn
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
        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device=self.betas.device, dtype=torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def calculate_v(self, x_start, t, noise=None):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = x_from.shape
        noise = default(noise, lambda : torch.randn_like(x_from))
        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)
        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05, fp16_eps=0.001, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class ChanLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05, fp16_eps=0.001, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, *, expansion_factor=2.0, depth=2, norm=False):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda : nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        layers = [nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn()))
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class RelPosBias(nn.Module):

    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class Attention(nn.Module):

    def __init__(self, dim, *, heads=8, dim_head=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(dim, mult=4):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult, bias=False), nn.GELU(), nn.Linear(dim * mult, dim, bias=False))


class CausalTransformer(nn.Module):

    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, norm_in=False, norm_out=True, attn_dropout=0.0, ff_dropout=0.0, final_proj=True, normformer=False, rotary_emb=True):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, causal=True, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_emb=rotary_emb), FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)]))
        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        out = self.norm(x)
        return self.project_out(out)


def is_float_dtype(dtype):
    return any([(dtype == float_dtype) for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), 'input to sinusoidal pos emb must be a float type'
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class DiffusionPriorNetwork(nn.Module):

    def __init__(self, dim, num_timesteps=None, num_time_embeds=1, num_image_embeds=1, num_text_embeds=1, max_text_len=256, self_cond=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds
        self.to_text_embeds = nn.Sequential(nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(), Rearrange('b (n d) -> b n d', n=num_text_embeds))
        self.continuous_embedded_time = not exists(num_timesteps)
        self.to_time_embeds = nn.Sequential(nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), Rearrange('b (n d) -> b n d', n=num_time_embeds))
        self.to_image_embeds = nn.Sequential(nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(), Rearrange('b (n d) -> b n d', n=num_image_embeds))
        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)
        self.max_text_len = max_text_len
        self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))
        self.self_cond = self_cond

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, text_cond_drop_prob=1.0, image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, image_embed, diffusion_timesteps, *, text_embed, text_encodings=None, self_cond=None, text_cond_drop_prob=0.0, image_cond_drop_prob=0.0):
        batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds
        if self.self_cond:
            self_cond = default(self_cond, lambda : torch.zeros(batch, self.dim, device=device, dtype=dtype))
            self_cond = rearrange(self_cond, 'b d -> b 1 d')
        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)
        text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob, device=device)
        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')
        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')
        if not exists(text_encodings):
            text_encodings = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        mask = torch.any(text_encodings != 0.0, dim=-1)
        text_encodings = text_encodings[:, :self.max_text_len]
        mask = mask[:, :self.max_text_len]
        text_len = text_encodings.shape[-2]
        remainder = self.max_text_len - text_len
        if remainder > 0:
            text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value=0.0)
            mask = F.pad(mask, (0, remainder), value=False)
        null_text_encodings = self.null_text_encodings
        text_encodings = torch.where(rearrange(mask, 'b n -> b n 1').clone() & text_keep_mask, text_encodings, null_text_encodings)
        null_text_embeds = self.null_text_embeds
        text_embed = torch.where(text_keep_mask, text_embed, null_text_embeds)
        null_image_embed = self.null_image_embed
        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed)
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)
        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b=batch)
        if self.self_cond:
            learned_queries = torch.cat((image_embed, self_cond), dim=-2)
        tokens = torch.cat((text_encodings, text_embed, time_embed, image_embed, learned_queries), dim=-2)
        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -1, :]
        return pred_image_embed


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


class DiffusionPrior(nn.Module):

    def __init__(self, net, *, clip=None, image_embed_dim=None, image_size=None, image_channels=3, timesteps=1000, sample_timesteps=None, cond_drop_prob=0.0, text_cond_drop_prob=None, image_cond_drop_prob=None, loss_type='l2', predict_x_start=True, predict_v=False, beta_schedule='cosine', condition_on_text_encodings=True, sampling_clamp_l2norm=False, sampling_final_clamp_l2norm=False, training_clamp_l2norm=False, init_image_embed_l2norm=False, image_embed_scale=None, clip_adapter_overrides=dict()):
        super().__init__()
        self.sample_timesteps = sample_timesteps
        self.noise_scheduler = NoiseScheduler(beta_schedule=beta_schedule, timesteps=timesteps, loss_type=loss_type)
        if exists(clip):
            assert image_channels == clip.image_channels, f'channels of image ({image_channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'
            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)
            assert isinstance(clip, BaseClipAdapter)
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            assert exists(image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
            self.clip = None
        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda : clip.dim_latent)
        assert net.dim == self.image_embed_dim, f'your diffusion prior network has a dimension of {net.dim}, but you set your image embedding dimension (keyword image_embed_dim) on DiffusionPrior to {self.image_embed_dim}'
        assert not exists(clip) or clip.dim_latent == self.image_embed_dim, f'you passed in a CLIP to the diffusion prior with latent dimensions of {clip.dim_latent}, but your image embedding dimension (keyword image_embed_dim) for the DiffusionPrior was set to {self.image_embed_dim}'
        self.channels = default(image_channels, lambda : clip.image_channels)
        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)
        self.can_classifier_guidance = self.text_cond_drop_prob > 0.0 and self.image_cond_drop_prob > 0.0
        self.condition_on_text_encodings = condition_on_text_encodings
        self.predict_x_start = predict_x_start
        self.predict_v = predict_v
        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm
        self.register_buffer('_dummy', torch.tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(self, x, t, text_cond, self_cond=None, clip_denoised=False, cond_scale=1.0):
        assert not (cond_scale != 1.0 and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'
        pred = self.net.forward_with_cond_scale(x, t, cond_scale=cond_scale, self_cond=self_cond, **text_cond)
        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1.0, 1.0)
        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale
        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, text_cond=text_cond, self_cond=self_cond, clip_denoised=clip_denoised, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0):
        batch, device = shape[0], self.device
        image_embed = torch.randn(shape, device=device)
        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond, cond_scale=cond_scale)
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape, text_cond, *, timesteps, eta=1.0, cond_scale=1.0):
        batch, device, alphas, total_timesteps = shape[0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps
        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        image_embed = torch.randn(shape, device=device)
        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha = alphas[time]
            alpha_next = alphas[time_next]
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            pred = self.net.forward_with_cond_scale(image_embed, time_cond, self_cond=self_cond, cond_scale=cond_scale, **text_cond)
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t=time_cond, noise=pred)
            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)
            if self.predict_x_start or self.predict_v:
                pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t=time_cond, x0=x_start)
            else:
                pred_noise = pred
            if time_next < 0:
                image_embed = x_start
                continue
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.0
            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)
        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps
        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)
        image_embed = normalized_image_embed / self.image_embed_scale
        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda : torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()
        pred = self.net(image_embed_noisy, times, self_cond=self_cond, text_cond_drop_prob=self.text_cond_drop_prob, image_cond_drop_prob=self.image_cond_drop_prob, **text_cond)
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)
        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise
        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale=1.0):
        device = self.betas.device
        shape = batch_size, self.image_embed_dim
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long), text_cond=text_cond, cond_scale=cond_scale)
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch=2, cond_scale=1.0, timesteps=None):
        timesteps = default(timesteps, self.sample_timesteps)
        text = repeat(text, 'b ... -> (b r) ...', r=num_samples_per_batch)
        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim
        text_embed, text_encodings = self.clip.embed_text(text)
        text_cond = dict(text_embed=text_embed)
        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}
        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale, timesteps=timesteps)
        text_embeds = text_cond['text_embed']
        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r=num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r=num_samples_per_batch)
        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k=1).indices
        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d=image_embed_dim)
        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(self, text=None, image=None, text_embed=None, image_embed=None, text_encodings=None, *args, **kwargs):
        assert exists(text) ^ exists(text_embed), 'either text or text embedding must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'
        if exists(image):
            image_embed, _ = self.clip.embed_image(image)
        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)
        text_cond = dict(text_embed=text_embed)
        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)
        image_embed *= self.image_embed_scale
        return self.p_losses(image_embed, times, *args, text_cond=text_cond, **kwargs)


class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        weight = self.weight
        flattened_weights = rearrange(weight, 'o ... -> o (...)')
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = torch.var(flattened_weights, dim=-1, unbiased=False)
        var = rearrange(var, 'o -> o 1 1 1')
        weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8, weight_standardization=False):
        super().__init__()
        conv_klass = nn.Conv2d if not weight_standardization else WeightStandardizedConv2d
        self.project = conv_klass(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.project(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, dropout=0.0, norm_context=False, cosine_sim=False, cosine_sim_scale=16):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        context = self.norm_context(context)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)
        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        if self.cosine_sim:
            q, k = map(l2norm, (q, k))
        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, cond_dim=None, time_cond_dim=None, groups=8, weight_standardization=False, cosine_sim_cross_attn=False):
        super().__init__()
        self.time_mlp = None
        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2))
        self.cross_attn = None
        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom('b c h w', 'b (h w) c', CrossAttention(dim=dim_out, context_dim=cond_dim, cosine_sim=cosine_sim_cross_attn))
        self.block1 = Block(dim, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.block2 = Block(dim_out, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, dim_head=32, heads=8, **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)
        self.nonlin = nn.GELU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim))

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        seq_len = x * y
        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h=h)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        v = l2norm(v)
        k, v = map(lambda t: t / math.sqrt(seq_len), (k, v))
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)
        out = self.nonlin(out)
        return self.to_out(out)


class CrossEmbedLayer(nn.Module):

    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: t % 2 == stride % 2, kernel_sizes)])
        dim_out = default(dim_out, dim_in)
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / 2 ** i) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class UpsampleCombiner(nn.Module):

    def __init__(self, dim, *, enabled=False, dim_ins=tuple(), dim_outs=tuple()):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.enabled = enabled
        if not self.enabled:
            self.dim_out = dim
            return
        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]
        fmaps = default(fmaps, tuple())
        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x
        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


def Downsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2), nn.Conv2d(dim * 4, dim_out, 1))


def NearestUpsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, dim_out, 3, padding=1))


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def identity(t, *args, **kwargs):
    return t


def maybe(fn):

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def make_checkpointable(fn, **kwargs):
    if isinstance(fn, nn.ModuleList):
        return [maybe(make_checkpointable)(el, **kwargs) for el in fn]
    condition = kwargs.pop('condition', None)
    if exists(condition) and not condition(fn):
        return fn

    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([(isinstance(el, torch.Tensor) and el.requires_grad) for el in args])
        if not input_needs_grad:
            return fn(*args)
        return checkpoint(fn, *args)
    return inner


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


class Unet(nn.Module):

    def __init__(self, dim, *, image_embed_dim=None, text_embed_dim=None, cond_dim=None, num_image_tokens=4, num_time_tokens=2, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, channels_out=None, self_attn=False, attn_dim_head=32, attn_heads=16, lowres_cond=False, lowres_noise_cond=False, self_cond=False, sparse_attn=False, cosine_sim_cross_attn=False, cosine_sim_self_attn=False, attend_at_middle=True, cond_on_text_encodings=False, max_text_len=256, cond_on_image_embeds=False, add_image_embeds_to_time=True, init_dim=None, init_conv_kernel_size=7, resnet_groups=8, resnet_weight_standardization=False, num_resnet_blocks=2, init_cross_embed=True, init_cross_embed_kernel_sizes=(3, 7, 15), cross_embed_downsample=False, cross_embed_downsample_kernel_sizes=(2, 4), memory_efficient=False, scale_skip_connection=False, pixel_shuffle_upsample=True, final_conv_kernel_size=1, combine_upsample_fmaps=False, checkpoint_during_training=False, **kwargs):
        super().__init__()
        self._locals = locals()
        del self._locals['self']
        del self._locals['__class__']
        self.lowres_cond = lowres_cond
        self.self_cond = self_cond
        self.channels = channels
        self.channels_out = default(channels_out, channels)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)
        self.init_conv = CrossEmbedLayer(init_channels, dim_out=init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_stages = len(in_out)
        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4
        self.to_time_hiddens = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), nn.GELU())
        self.to_time_tokens = nn.Sequential(nn.Linear(time_cond_dim, cond_dim * num_time_tokens), Rearrange('b (r d) -> b r d', r=num_time_tokens))
        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))
        self.image_to_tokens = nn.Sequential(nn.Linear(image_embed_dim, cond_dim * num_image_tokens), Rearrange('b (n d) -> b n d', n=num_image_tokens)) if cond_on_image_embeds and image_embed_dim != cond_dim else nn.Identity()
        self.to_image_hiddens = nn.Sequential(nn.Linear(image_embed_dim, time_cond_dim), nn.GELU()) if cond_on_image_embeds and add_image_embeds_to_time else None
        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)
        self.text_to_cond = None
        self.text_embed_dim = None
        if cond_on_text_encodings:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text_encodings is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
            self.text_embed_dim = text_embed_dim
        self.lowres_noise_cond = lowres_noise_cond
        self.to_lowres_noise_cond = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), nn.GELU(), nn.Linear(time_cond_dim, time_cond_dim)) if lowres_noise_cond else None
        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds
        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_image_hiddens = nn.Parameter(torch.randn(1, time_cond_dim))
        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.skip_connect_scale = 1.0 if not scale_skip_connection else 2 ** -0.5
        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head, cosine_sim=cosine_sim_self_attn)
        self_attn = cast_tuple(self_attn, num_stages)
        create_self_attn = lambda dim: EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(dim, **attn_kwargs)))
        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)
        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes)
        upsample_klass = NearestUpsample if not pixel_shuffle_upsample else PixelShuffleUpsample
        resnet_block = partial(ResnetBlock, cosine_sim_cross_attn=cosine_sim_cross_attn, weight_standardization=resnet_weight_standardization)
        self.init_resnet_block = resnet_block(init_dim, init_dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group) if memory_efficient else None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        skip_connect_dims = []
        upsample_combiner_dims = []
        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= num_resolutions - 1
            layer_cond_dim = cond_dim if not is_first else None
            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)
            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))
            self.downs.append(nn.ModuleList([downsample_klass(dim_in, dim_out=dim_out) if memory_efficient else None, resnet_block(dim_layer, dim_layer, time_cond_dim=time_cond_dim, groups=groups), nn.ModuleList([resnet_block(dim_layer, dim_layer, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim, groups=groups) for _ in range(layer_num_resnet_blocks)]), attention, downsample_klass(dim_layer, dim_out=dim_out) if not is_last and not memory_efficient else nn.Conv2d(dim_layer, dim_out, 1)]))
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim, groups=resnet_groups[-1])
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim, groups=resnet_groups[-1])
        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(reversed(in_out), reversed(resnet_groups), reversed(num_resnet_blocks), reversed(self_attn))):
            is_last = ind >= len(in_out) - 1
            layer_cond_dim = cond_dim if not is_last else None
            skip_connect_dim = skip_connect_dims.pop()
            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))
            upsample_combiner_dims.append(dim_out)
            self.ups.append(nn.ModuleList([resnet_block(dim_out + skip_connect_dim, dim_out, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim, groups=groups), nn.ModuleList([resnet_block(dim_out + skip_connect_dim, dim_out, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim, groups=groups) for _ in range(layer_num_resnet_blocks)]), attention, upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else nn.Identity()]))
        self.upsample_combiner = UpsampleCombiner(dim=dim, enabled=combine_upsample_fmaps, dim_ins=upsample_combiner_dims, dim_outs=(dim,) * len(upsample_combiner_dims))
        self.final_resnet_block = resnet_block(self.upsample_combiner.dim_out + dim, dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group)
        out_dim_in = dim + (channels if lowres_cond else 0)
        self.to_out = nn.Conv2d(out_dim_in, self.channels_out, kernel_size=final_conv_kernel_size, padding=final_conv_kernel_size // 2)
        zero_init_(self.to_out)
        self.checkpoint_during_training = checkpoint_during_training

    def cast_model_parameters(self, *, lowres_cond, lowres_noise_cond, channels, channels_out, cond_on_image_embeds, cond_on_text_encodings):
        if lowres_cond == self.lowres_cond and channels == self.channels and cond_on_image_embeds == self.cond_on_image_embeds and cond_on_text_encodings == self.cond_on_text_encodings and lowres_noise_cond == self.lowres_noise_cond and channels_out == self.channels_out:
            return self
        updated_kwargs = dict(lowres_cond=lowres_cond, channels=channels, channels_out=channels_out, cond_on_image_embeds=cond_on_image_embeds, cond_on_text_encodings=cond_on_text_encodings, lowres_noise_cond=lowres_noise_cond)
        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, text_cond_drop_prob=1.0, image_cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, *, image_embed, lowres_cond_img=None, lowres_noise_level=None, text_encodings=None, image_cond_drop_prob=0.0, text_cond_drop_prob=0.0, blur_sigma=None, blur_kernel_size=None, disable_checkpoint=False, self_cond=None):
        batch_size, device = x.shape[0], x.device
        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        if self.self_cond:
            self_cond = default(self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim=1)
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        time = time.type_as(x)
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)
        if exists(lowres_noise_level):
            assert exists(self.to_lowres_noise_cond), 'lowres_noise_cond must be set to True on instantiation of the unet in order to conditiong on lowres noise'
            lowres_noise_level = lowres_noise_level.type_as(x)
            t = t + self.to_lowres_noise_cond(lowres_noise_level)
        image_keep_mask = prob_mask_like((batch_size,), 1 - image_cond_drop_prob, device=device)
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob, device=device)
        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')
        if exists(image_embed) and exists(self.to_image_hiddens):
            image_hiddens = self.to_image_hiddens(image_embed)
            image_keep_mask_hidden = rearrange(image_keep_mask, 'b -> b 1')
            null_image_hiddens = self.null_image_hiddens
            image_hiddens = torch.where(image_keep_mask_hidden, image_hiddens, null_image_hiddens)
            t = t + image_hiddens
        image_tokens = None
        if self.cond_on_image_embeds:
            image_keep_mask_embed = rearrange(image_keep_mask, 'b -> b 1 1')
            image_tokens = self.image_to_tokens(image_embed)
            null_image_embed = self.null_image_embed
            image_tokens = torch.where(image_keep_mask_embed, image_tokens, null_image_embed)
        text_tokens = None
        if exists(text_encodings) and self.cond_on_text_encodings:
            assert text_encodings.shape[0] == batch_size, f'the text encodings being passed into the unet does not have the proper batch size - text encoding shape {text_encodings.shape} - required batch size is {batch_size}'
            assert self.text_embed_dim == text_encodings.shape[-1], f'the text encodings you are passing in have a dimension of {text_encodings.shape[-1]}, but the unet was created with text_embed_dim of {self.text_embed_dim}.'
            text_mask = torch.any(text_encodings != 0.0, dim=-1)
            text_tokens = self.text_to_cond(text_encodings)
            text_tokens = text_tokens[:, :self.max_text_len]
            text_mask = text_mask[:, :self.max_text_len]
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                text_mask = F.pad(text_mask, (0, remainder), value=False)
            text_mask = rearrange(text_mask, 'b n -> b n 1')
            assert text_mask.shape[0] == text_keep_mask.shape[0], f'text_mask has shape of {text_mask.shape} while text_keep_mask has shape {text_keep_mask.shape}. text encoding is of shape {text_encodings.shape}'
            text_keep_mask = text_mask & text_keep_mask
            null_text_embed = self.null_text_embed
            text_tokens = torch.where(text_keep_mask, text_tokens, null_text_embed)
        c = time_tokens
        if exists(image_tokens):
            c = torch.cat((c, image_tokens), dim=-2)
        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim=-2)
        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)
        can_checkpoint = self.training and self.checkpoint_during_training and not disable_checkpoint
        apply_checkpoint_fn = make_checkpointable if can_checkpoint else identity
        init_resnet_block, mid_block1, mid_attn, mid_block2, final_resnet_block = [maybe(apply_checkpoint_fn)(module) for module in (self.init_resnet_block, self.mid_block1, self.mid_attn, self.mid_block2, self.final_resnet_block)]
        can_checkpoint_cond = lambda m: isinstance(m, ResnetBlock)
        downs, ups = [maybe(apply_checkpoint_fn)(m, condition=can_checkpoint_cond) for m in (self.downs, self.ups)]
        if exists(init_resnet_block):
            x = init_resnet_block(x, t)
        down_hiddens = []
        up_hiddens = []
        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in downs:
            if exists(pre_downsample):
                x = pre_downsample(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                down_hiddens.append(x.contiguous())
            x = attn(x)
            down_hiddens.append(x.contiguous())
            if exists(post_downsample):
                x = post_downsample(x)
        x = mid_block1(x, t, mid_c)
        if exists(mid_attn):
            x = mid_attn(x)
        x = mid_block2(x, t, mid_c)
        connect_skip = lambda fmap: torch.cat((fmap, down_hiddens.pop() * self.skip_connect_scale), dim=1)
        for init_block, resnet_blocks, attn, upsample in ups:
            x = connect_skip(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)
            x = attn(x)
            up_hiddens.append(x.contiguous())
            x = upsample(x)
        x = self.upsample_combiner(x, up_hiddens)
        x = torch.cat((x, r), dim=1)
        x = final_resnet_block(x, t)
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)
        return self.to_out(x)


class LowresConditioner(nn.Module):

    def __init__(self, downsample_first=True, use_blur=True, blur_prob=0.5, blur_sigma=0.6, blur_kernel_size=3, use_noise=False, input_image_range=None, normalize_img_fn=identity, unnormalize_img_fn=identity):
        super().__init__()
        self.downsample_first = downsample_first
        self.input_image_range = input_image_range
        self.use_blur = use_blur
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size
        self.use_noise = use_noise
        self.normalize_img = normalize_img_fn
        self.unnormalize_img = unnormalize_img_fn
        self.noise_scheduler = NoiseScheduler(beta_schedule='linear', timesteps=1000, loss_type='l2') if use_noise else None

    def noise_image(self, cond_fmap, noise_levels=None):
        assert exists(self.noise_scheduler)
        batch = cond_fmap.shape[0]
        cond_fmap = self.normalize_img(cond_fmap)
        random_noise_levels = default(noise_levels, lambda : self.noise_scheduler.sample_random_times(batch))
        cond_fmap = self.noise_scheduler.q_sample(cond_fmap, t=random_noise_levels, noise=torch.randn_like(cond_fmap))
        cond_fmap = self.unnormalize_img(cond_fmap)
        return cond_fmap, random_noise_levels

    def forward(self, cond_fmap, *, target_image_size, downsample_image_size=None, should_blur=True, blur_sigma=None, blur_kernel_size=None):
        if self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size, clamp_range=self.input_image_range, nearest=True)
        if self.use_blur and should_blur and random.random() < self.blur_prob:
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)
            if isinstance(blur_sigma, tuple):
                blur_sigma = tuple(map(float, blur_sigma))
                blur_sigma = random.uniform(*blur_sigma)
            if isinstance(blur_kernel_size, tuple):
                blur_kernel_size = tuple(map(int, blur_kernel_size))
                kernel_size_lo, kernel_size_hi = blur_kernel_size
                blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)
            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))
        cond_fmap = resize_image_to(cond_fmap, target_image_size, clamp_range=self.input_image_range, nearest=True)
        random_noise_levels = None
        if self.use_noise:
            cond_fmap, random_noise_levels = self.noise_image(cond_fmap)
        return cond_fmap, random_noise_levels


NAT = 1.0 / math.log(2.0)


class NullVQGanVAE(nn.Module):

    def __init__(self, *, channels):
        super().__init__()
        self.encoded_dim = channels
        self.layers = 0

    def get_encoded_fmap_size(self, size):
        return size

    def copy_for_eval(self):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x


UnetOutput = namedtuple('UnetOutput', ['pred', 'var_interp_frac_unnormalized'])


MList = nn.ModuleList


def leaky_relu(p=0.1):
    return nn.LeakyReLU(0.1)


class Discriminator(nn.Module):

    def __init__(self, dims, channels=3, groups=16, init_kernel_size=5):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])
        self.layers = MList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding=init_kernel_size // 2), leaky_relu())])
        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1), nn.GroupNorm(groups, dim_out), leaky_relu()))
        dim = dims[-1]
        self.to_logits = nn.Sequential(nn.Conv2d(dim, dim, 1), leaky_relu(), nn.Conv2d(dim, 1, 4))

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return self.to_logits(x)


class GLUResBlock(nn.Module):

    def __init__(self, chan, groups=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan * 2, 3, padding=1), nn.GLU(dim=1), nn.GroupNorm(groups, chan), nn.Conv2d(chan, chan * 2, 3, padding=1), nn.GLU(dim=1), nn.GroupNorm(groups, chan), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


class ResBlock(nn.Module):

    def __init__(self, chan, groups=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan, 3, padding=1), nn.GroupNorm(groups, chan), leaky_relu(), nn.Conv2d(chan, chan, 3, padding=1), nn.GroupNorm(groups, chan), leaky_relu(), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers=2):
        super().__init__()
        self.net = MList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))
        self.net.append(nn.Linear(dim, heads))
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))
        if not exists(self.rel_pos):
            pos = torch.arange(fmap_size, device=device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer('rel_pos', rel_pos, persistent=False)
        rel_pos = self.rel_pos.float()
        for layer in self.net:
            rel_pos = layer(rel_pos)
        bias = rearrange(rel_pos, 'i j h -> h i j')
        return x + bias


class LayerNormChan(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


def stable_softmax(t, dim=-1, alpha=32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


class VQGanAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = LayerNormChan(dim)
        self.cpb = ContinuousPositionBias(dim=dim // 4, heads=heads)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()
        x = self.pre_norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=h), (q, k, v))
        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale
        sim = self.cpb(sim)
        attn = stable_softmax(sim, dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=height, y=width)
        out = self.to_out(out)
        return out + residual


class ResnetEncDec(nn.Module):

    def __init__(self, dim, *, channels=3, layers=4, layer_mults=None, num_resnet_blocks=1, resnet_groups=16, first_conv_kernel_size=5, use_attn=True, attn_dim_head=64, attn_heads=8, attn_dropout=0.0):
        super().__init__()
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'
        self.layers = layers
        self.encoders = MList([])
        self.decoders = MList([])
        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'
        layer_dims = [(dim * mult) for mult in layer_mults]
        dims = dim, *layer_dims
        self.encoded_dim = dims[-1]
        dim_pairs = zip(dims[:-1], dims[1:])
        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)
        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = *((0,) * (layers - 1)), num_resnet_blocks
        if not isinstance(use_attn, tuple):
            use_attn = *((False,) * (layers - 1)), use_attn
        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'
        assert len(use_attn) == layers
        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks, layer_use_attn in zip(range(layers), dim_pairs, num_resnet_blocks, use_attn):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))
            if layer_use_attn:
                prepend(self.decoders, VQGanAttention(dim=dim_out, heads=attn_heads, dim_head=attn_dim_head, dropout=attn_dropout))
            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups=resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups=resnet_groups))
            if layer_use_attn:
                append(self.encoders, VQGanAttention(dim=dim_out, heads=attn_heads, dim_head=attn_dim_head, dropout=attn_dropout))
        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding=first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    def get_encoded_fmap_size(self, image_size):
        return image_size // 2 ** self.layers

    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    def encode(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

    def decode(self, x):
        for dec in self.decoders:
            x = dec(x)
        return x


class RearrangeImage(nn.Module):

    def forward(self, x):
        n = x.shape[1]
        w = h = int(sqrt(n))
        return rearrange(x, 'b (h w) ... -> b h w ...', h=h, w=w)


class Transformer(nn.Module):

    def __init__(self, dim, *, layers, dim_head=32, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTEncDec(nn.Module):

    def __init__(self, dim, channels=3, layers=4, patch_size=8, dim_head=32, heads=8, ff_mult=4):
        super().__init__()
        self.encoded_dim = dim
        self.patch_size = patch_size
        input_dim = channels * patch_size ** 2
        self.encoder = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(input_dim, dim), Transformer(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, layers=layers), RearrangeImage(), Rearrange('b h w c -> b c h w'))
        self.decoder = nn.Sequential(Rearrange('b c h w -> b (h w) c'), Transformer(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, layers=layers), nn.Sequential(nn.Linear(dim, dim * 4, bias=False), nn.Tanh(), nn.Linear(dim * 4, input_dim, bias=False)), RearrangeImage(), Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size))

    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    @property
    def last_dec_layer(self):
        return self.decoder[-3][-1].weight

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def log(t, eps=1e-10):
    return torch.log(t + eps)


def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()


def grad_layer_wrt_loss(loss, layer):
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=images.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def remove_vgg(fn):

    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')
        out = fn(self, *args, **kwargs)
        if has_vgg:
            self.vgg = vgg
        return out
    return inner


def safe_div(numer, denom, eps=1e-08):
    return numer / (denom + eps)


class VQGanVAE(nn.Module):

    def __init__(self, *, dim, image_size, channels=3, layers=4, l2_recon_loss=False, use_hinge_loss=True, vgg=None, vq_codebook_dim=256, vq_codebook_size=512, vq_decay=0.8, vq_commitment_weight=1.0, vq_kmeans_init=True, vq_use_cosine_sim=True, use_vgg_and_gan=True, vae_type='resnet', discr_layers=4, **kwargs):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)
        self.image_size = image_size
        self.channels = channels
        self.codebook_size = vq_codebook_size
        if vae_type == 'resnet':
            enc_dec_klass = ResnetEncDec
        elif vae_type == 'vit':
            enc_dec_klass = ViTEncDec
        else:
            raise ValueError(f'{vae_type} not valid')
        self.enc_dec = enc_dec_klass(dim=dim, channels=channels, layers=layers, **encdec_kwargs)
        self.vq = VQ(dim=self.enc_dec.encoded_dim, codebook_dim=vq_codebook_dim, codebook_size=vq_codebook_size, decay=vq_decay, commitment_weight=vq_commitment_weight, accept_image_fmap=True, kmeans_init=vq_kmeans_init, use_cosine_sim=vq_use_cosine_sim, **vq_kwargs)
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss
        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan
        if not use_vgg_and_gan:
            return
        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained=True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [(dim * mult) for mult in layer_mults]
        dims = dim, *layer_dims
        self.discr = Discriminator(dims=dims, channels=channels)
        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())
        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg
        vae_copy.eval()
        return vae_copy

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        return fmap

    def decode(self, fmap, return_indices_and_loss=False):
        fmap, indices, commit_loss = self.vq(fmap)
        fmap = self.enc_dec.decode(fmap)
        if not return_indices_and_loss:
            return fmap
        return fmap, indices, commit_loss

    def forward(self, img, return_loss=False, return_discr_loss=False, return_recons=False, add_gradient_penalty=True):
        batch, channels, height, width, device = *img.shape, img.device
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'
        fmap = self.encode(img)
        fmap, indices, commit_loss = self.decode(fmap, return_indices_and_loss=True)
        if not return_loss and not return_discr_loss:
            return fmap
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'
            fmap.detach_()
            img.requires_grad_()
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)
            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp
            if return_recons:
                return loss, fmap
            return loss
        recon_loss = self.recon_loss_fn(fmap, img)
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap
            return recon_loss
        img_vgg_input = img
        fmap_vgg_input = fmap
        if img.shape[1] == 1:
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3), (img_vgg_input, fmap_vgg_input))
        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        gen_loss = self.gen_loss(self.discr(fmap))
        last_dec_layer = self.enc_dec.last_dec_layer
        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)
        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max=10000.0)
        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss
        if return_recons:
            return loss, fmap
        return loss


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape
    eps = 1e-12 if x.dtype == torch.float32 else 0.001
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps=eps)
    log_one_minus_cdf_min = log(1.0 - cdf_min, eps=eps)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -thres, log_cdf_plus, torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta, eps=eps)))
    return log_probs


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def module_device(module):
    if isinstance(module, nn.Identity):
        return 'cpu'
    return next(module.parameters()).device


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return *t, *((fillvalue,) * remain_length)


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([(type(el) == str) for el in x])


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/bpe_simple_vocab_16e6.txt')


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def whitespace_clean(text):
    text = re.sub('\\s+', ' ', text)
    text = text.strip()
    return text


class DALLE2(nn.Module):

    def __init__(self, *, prior, decoder, prior_num_samples=2):
        super().__init__()
        assert isinstance(prior, DiffusionPrior)
        assert isinstance(decoder, Decoder)
        self.prior = prior
        self.decoder = decoder
        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings
        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    @eval_decorator
    def forward(self, text, cond_scale=1.0, prior_cond_scale=1.0, return_pil_images=False):
        device = module_device(self)
        one_text = isinstance(text, str) or not is_list_str(text) and text.shape[0] == 1
        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text)
        image_embed = self.prior.sample(text, num_samples_per_batch=self.prior_num_samples, cond_scale=prior_cond_scale)
        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed=image_embed, text=text_cond, cond_scale=cond_scale)
        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim=0)))
        if one_text:
            return first(images)
        return images


def cast_torch_tensor(fn):

    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', next(model.parameters()).device)
        cast_device = kwargs.pop('_cast_device', True)
        cast_deepspeed_precision = kwargs.pop('_cast_deepspeed_precision', True)
        kwargs_keys = kwargs.keys()
        all_args = *args, *kwargs.values()
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))
        if cast_device:
            all_args = tuple(map(lambda t: t if exists(t) and isinstance(t, torch.Tensor) else t, all_args))
        if cast_deepspeed_precision:
            try:
                accelerator = model.accelerator
                if accelerator is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
                    cast_type_map = {'fp16': torch.half, 'bf16': torch.bfloat16, 'no': torch.float}
                    precision_type = cast_type_map[accelerator.mixed_precision]
                    all_args = tuple(map(lambda t: t if exists(t) and isinstance(t, torch.Tensor) else t, all_args))
            except AttributeError:
                pass
        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))
        out = fn(model, *args, **kwargs)
        return out
    return inner


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=False, group_wd_params=True, **kwargs):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)
    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None


def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index:start_index + split_size])
    return accum


def split(t, split_size=None):
    if not exists(split_size):
        return t
    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim=0)
    if isinstance(t, Iterable):
        return split_iterable(t, split_size)
    return TypeError


def split_args_and_kwargs(*args, split_size=None, **kwargs):
    all_args = *args, *kwargs.values()
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)
    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)
    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len
    split_all_args = [(split(arg, split_size=split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else (arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))
    for chunk_size, *chunked_all_args in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)


def prior_sample_in_chunks(fn):

    @wraps(fn)
    def inner(self, *args, max_batch_size=None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)
        outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs)]
        return torch.cat(outputs, dim=0)
    return inner


class DiffusionPriorTrainer(nn.Module):

    def __init__(self, diffusion_prior, accelerator=None, use_ema=True, lr=0.0003, wd=0.01, eps=1e-06, max_grad_norm=None, group_wd_params=True, warmup_steps=None, cosine_decay_max_steps=None, **kwargs):
        super().__init__()
        assert isinstance(diffusion_prior, DiffusionPrior)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        accelerator_kwargs, kwargs = groupby_prefix_and_trim('accelerator_', kwargs)
        if not exists(accelerator):
            accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator = accelerator
        self.text_conditioned = diffusion_prior.condition_on_text_encodings
        self.device = accelerator.device
        diffusion_prior
        self.diffusion_prior = diffusion_prior
        if exists(self.accelerator) and self.accelerator.distributed_type == DistributedType.DEEPSPEED and self.diffusion_prior.clip is not None:
            cast_type_map = {'fp16': torch.half, 'bf16': torch.bfloat16, 'no': torch.float}
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, 'DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip'
            self.diffusion_prior.clip
        self.optim_kwargs = dict(lr=lr, wd=wd, eps=eps, group_wd_params=group_wd_params)
        self.optimizer = get_optimizer(self.diffusion_prior.parameters(), **self.optim_kwargs, **kwargs)
        if exists(cosine_decay_max_steps):
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_decay_max_steps)
        else:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=warmup_steps) if exists(warmup_steps) else None
        self.diffusion_prior, self.optimizer, self.scheduler = self.accelerator.prepare(self.diffusion_prior, self.optimizer, self.scheduler)
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion_prior = EMA(self.accelerator.unwrap_model(self.diffusion_prior), **ema_kwargs)
        self.max_grad_norm = max_grad_norm
        self.register_buffer('step', torch.tensor([0], device=self.device))

    def save(self, path, overwrite=True, **kwargs):
        if self.accelerator.is_main_process:
            None
            path = Path(path)
            assert not (path.exists() and not overwrite)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_obj = dict(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), warmup_scheduler=self.warmup_scheduler, model=self.accelerator.unwrap_model(self.diffusion_prior).state_dict(), version=version.parse(__version__), step=self.step, **kwargs)
            if self.use_ema:
                save_obj = {**save_obj, 'ema': self.ema_diffusion_prior.state_dict(), 'ema_model': self.ema_diffusion_prior.ema_model.state_dict()}
            torch.save(save_obj, str(path))

    def load(self, path_or_state, overwrite_lr=True, strict=True):
        """
        Load a checkpoint of a diffusion prior trainer.

        Will load the entire trainer, including the optimizer and EMA.

        Params:
            - path_or_state (str | torch): a path to the DiffusionPriorTrainer checkpoint file
            - overwrite_lr (bool): wether or not to overwrite the stored LR with the LR specified in the new trainer
            - strict (bool): kwarg for `torch.nn.Module.load_state_dict`, will force an exact checkpoint match

        Returns:
            loaded_obj (dict): The loaded checkpoint dictionary
        """
        if isinstance(path_or_state, str):
            path = Path(path_or_state)
            assert path.exists()
            loaded_obj = torch.load(str(path), map_location=self.device)
        elif isinstance(path_or_state, dict):
            loaded_obj = path_or_state
        if version.parse(__version__) != loaded_obj['version']:
            None
        self.accelerator.unwrap_model(self.diffusion_prior).load_state_dict(loaded_obj['model'], strict=strict)
        self.step.copy_(torch.ones_like(self.step, device=self.device) * loaded_obj['step'])
        self.optimizer.load_state_dict(loaded_obj['optimizer'])
        self.scheduler.load_state_dict(loaded_obj['scheduler'])
        if exists(self.warmup_scheduler):
            self.warmup_scheduler.last_step = self.step.item()
        if overwrite_lr:
            new_lr = self.optim_kwargs['lr']
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr if group['lr'] > 0.0 else 0.0
        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_diffusion_prior.load_state_dict(loaded_obj['ema'], strict=strict)
            self.ema_diffusion_prior.ema_model.load_state_dict(loaded_obj['ema_model'])
        return loaded_obj

    def update(self):
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.diffusion_prior.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if not self.accelerator.optimizer_step_was_skipped:
            sched_context = self.warmup_scheduler.dampening if exists(self.warmup_scheduler) else nullcontext
            with sched_context():
                self.scheduler.step()
        if self.use_ema:
            self.ema_diffusion_prior.update()
        self.step += 1

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def p_sample_loop(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.p_sample_loop(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def sample(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_batch_size(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample_batch_size(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.diffusion_prior).clip.embed_text(*args, **kwargs)

    @cast_torch_tensor
    def forward(self, *args, max_batch_size=None, **kwargs):
        total_loss = 0.0
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs):
            with self.accelerator.autocast():
                loss = self.diffusion_prior(*chunked_args, **chunked_kwargs)
                loss = loss * chunk_size_frac
            total_loss += loss.item()
            if self.training:
                self.accelerator.backward(loss)
        return total_loss


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def decoder_sample_in_chunks(fn):

    @wraps(fn)
    def inner(self, *args, max_batch_size=None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)
        if self.decoder.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs)]
        return torch.cat(outputs, dim=0)
    return inner


class DecoderTrainer(nn.Module):

    def __init__(self, decoder, accelerator=None, dataloaders=None, use_ema=True, lr=0.0001, wd=0.01, eps=1e-08, warmup_steps=None, cosine_decay_max_steps=None, max_grad_norm=0.5, amp=False, group_wd_params=True, **kwargs):
        super().__init__()
        assert isinstance(decoder, Decoder)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        self.accelerator = default(accelerator, Accelerator)
        self.num_unets = len(decoder.unets)
        self.use_ema = use_ema
        self.ema_unets = nn.ModuleList([])
        self.amp = amp
        lr, wd, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length=self.num_unets), (lr, wd, eps, warmup_steps, cosine_decay_max_steps))
        assert all([(unet_lr <= 0.01) for unet_lr in lr]), 'your learning rate is too high, recommend sticking with 1e-4, at most 5e-4'
        optimizers = []
        schedulers = []
        warmup_schedulers = []
        for unet, unet_lr, unet_wd, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps in zip(decoder.unets, lr, wd, eps, warmup_steps, cosine_decay_max_steps):
            if isinstance(unet, nn.Identity):
                optimizers.append(None)
                schedulers.append(None)
                warmup_schedulers.append(None)
            else:
                optimizer = get_optimizer(unet.parameters(), lr=unet_lr, wd=unet_wd, eps=unet_eps, group_wd_params=group_wd_params, **kwargs)
                optimizers.append(optimizer)
                if exists(unet_cosine_decay_max_steps):
                    scheduler = CosineAnnealingLR(optimizer, T_max=unet_cosine_decay_max_steps)
                else:
                    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=unet_warmup_steps) if exists(unet_warmup_steps) else None
                warmup_schedulers.append(warmup_scheduler)
                schedulers.append(scheduler)
            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))
        self.max_grad_norm = max_grad_norm
        self.register_buffer('steps', torch.tensor([0] * self.num_unets))
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and decoder.clip is not None:
            cast_type_map = {'fp16': torch.half, 'bf16': torch.bfloat16, 'no': torch.float}
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, 'DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip'
            clip = decoder.clip
            clip
        decoder, *optimizers = list(self.accelerator.prepare(decoder, *optimizers))
        self.decoder = decoder
        train_loader = val_loader = None
        if exists(dataloaders):
            train_loader, val_loader = self.accelerator.prepare(dataloaders['train'], dataloaders['val'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        for opt_ind, optimizer in zip(range(len(optimizers)), optimizers):
            setattr(self, f'optim{opt_ind}', optimizer)
        for sched_ind, scheduler in zip(range(len(schedulers)), schedulers):
            setattr(self, f'sched{sched_ind}', scheduler)
        self.warmup_schedulers = warmup_schedulers

    def validate_and_return_unet_number(self, unet_number=None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)
        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
        return unet_number

    def num_steps_taken(self, unet_number=None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        return self.steps[unet_number - 1].item()

    def save(self, path, overwrite=True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_obj = dict(model=self.accelerator.unwrap_model(self.decoder).state_dict(), version=__version__, steps=self.steps.cpu(), **kwargs)
        for ind in range(0, self.num_unets):
            optimizer_key = f'optim{ind}'
            scheduler_key = f'sched{ind}'
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            optimizer_state_dict = optimizer.state_dict() if exists(optimizer) else None
            scheduler_state_dict = scheduler.state_dict() if exists(scheduler) else None
            save_obj = {**save_obj, optimizer_key: optimizer_state_dict, scheduler_key: scheduler_state_dict}
        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}
        self.accelerator.save(save_obj, str(path))

    def load_state_dict(self, loaded_obj, only_model=False, strict=True):
        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.accelerator.print(f"loading saved decoder at version {loaded_obj['version']}, but current package version is {__version__}")
        self.accelerator.unwrap_model(self.decoder).load_state_dict(loaded_obj['model'], strict=strict)
        self.steps.copy_(loaded_obj['steps'])
        if only_model:
            return loaded_obj
        for ind, last_step in zip(range(0, self.num_unets), self.steps.tolist()):
            optimizer_key = f'optim{ind}'
            optimizer = getattr(self, optimizer_key)
            scheduler_key = f'sched{ind}'
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = self.warmup_schedulers[ind]
            if exists(optimizer):
                optimizer.load_state_dict(loaded_obj[optimizer_key])
            if exists(scheduler):
                scheduler.load_state_dict(loaded_obj[scheduler_key])
            if exists(warmup_scheduler):
                warmup_scheduler.last_step = last_step
        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj['ema'], strict=strict)

    def load(self, path, only_model=False, strict=True):
        path = Path(path)
        assert path.exists()
        loaded_obj = torch.load(str(path), map_location='cpu')
        self.load_state_dict(loaded_obj, only_model=only_model, strict=strict)
        return loaded_obj

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def increment_step(self, unet_number):
        assert 1 <= unet_number <= self.num_unets
        unet_index_tensor = torch.tensor(unet_number - 1, device=self.steps.device)
        self.steps += F.one_hot(unet_index_tensor, num_classes=len(self.steps))

    def update(self, unet_number=None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        index = unet_number - 1
        optimizer = getattr(self, f'optim{index}')
        scheduler = getattr(self, f'sched{index}')
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        warmup_scheduler = self.warmup_schedulers[index]
        scheduler_context = warmup_scheduler.dampening if exists(warmup_scheduler) else nullcontext
        with scheduler_context():
            scheduler.step()
        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()
        self.increment_step(unet_number)

    @torch.no_grad()
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        distributed = self.accelerator.num_processes > 1
        base_decoder = self.accelerator.unwrap_model(self.decoder)
        was_training = base_decoder.training
        base_decoder.eval()
        if kwargs.pop('use_non_ema', False) or not self.use_ema:
            out = base_decoder.sample(*args, **kwargs, distributed=distributed)
            base_decoder.train(was_training)
            return out
        trainable_unets = self.accelerator.unwrap_model(self.decoder).unets
        base_decoder.unets = self.unets
        output = base_decoder.sample(*args, **kwargs, distributed=distributed)
        base_decoder.unets = trainable_unets
        for ema in self.ema_unets:
            ema.restore_ema_model_device()
        base_decoder.train(was_training)
        return output

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_text(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_image(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_image(*args, **kwargs)

    @cast_torch_tensor
    def forward(self, *args, unet_number=None, max_batch_size=None, return_lowres_cond_image=False, **kwargs):
        unet_number = self.validate_and_return_unet_number(unet_number)
        total_loss = 0.0
        cond_images = []
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs):
            with self.accelerator.autocast():
                loss_obj = self.decoder(*chunked_args, unet_number=unet_number, return_lowres_cond_image=return_lowres_cond_image, **chunked_kwargs)
                if return_lowres_cond_image:
                    loss, cond_image = loss_obj
                else:
                    loss = loss_obj
                    cond_image = None
                loss = loss * chunk_size_frac
                if cond_image is not None:
                    cond_images.append(cond_image)
            total_loss += loss.item()
            if self.training:
                self.accelerator.backward(loss)
        if return_lowres_cond_image:
            return total_loss, torch.stack(cond_images)
        else:
            return total_loss


class ImageDataset(Dataset):

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        None
        self.transform = T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), T.Resize(image_size), T.RandomHorizontalFlip(), T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def cycle(dl):
    while True:
        for data in dl:
            yield data


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


class VQGanVAETrainer(nn.Module):

    def __init__(self, vae, *, num_train_steps, lr, batch_size, folder, grad_accum_every, wd=0.0, save_results_every=100, save_model_every=1000, results_folder='./results', valid_frac=0.05, random_split_seed=42, ema_beta=0.995, ema_update_after_step=500, ema_update_every=10, apply_grad_penalty_every=4, amp=False):
        super().__init__()
        assert isinstance(vae, VQGanVAE), 'vae must be instance of VQGanVAE'
        image_size = vae.image_size
        self.vae = vae
        self.ema_vae = EMA(vae, update_after_step=ema_update_after_step, update_every=ema_update_every)
        self.register_buffer('steps', torch.Tensor([0]))
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters
        self.optim = get_optimizer(vae_parameters, lr=lr, wd=wd)
        self.discr_optim = get_optimizer(discr_parameters, lr=lr, wd=wd)
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.discr_scaler = GradScaler(enabled=amp)
        self.ds = ImageDataset(folder, image_size=image_size)
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            None
        else:
            self.valid_ds = self.ds
            None
        self.dl = cycle(DataLoader(self.ds, batch_size=batch_size, shuffle=True))
        self.valid_dl = cycle(DataLoader(self.valid_ds, batch_size=batch_size, shuffle=True))
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.apply_grad_penalty_every = apply_grad_penalty_every
        self.results_folder = Path(results_folder)
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))
        self.results_folder.mkdir(parents=True, exist_ok=True)

    def train_step(self):
        device = next(self.vae.parameters()).device
        steps = int(self.steps.item())
        apply_grad_penalty = not steps % self.apply_grad_penalty_every
        self.vae.train()
        logs = {}
        for _ in range(self.grad_accum_every):
            img = next(self.dl)
            img = img
            with autocast(enabled=self.amp):
                loss = self.vae(img, return_loss=True, apply_grad_penalty=apply_grad_penalty)
                self.scaler.scale(loss / self.grad_accum_every).backward()
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad()
        if exists(self.vae.discr):
            discr_loss = 0
            for _ in range(self.grad_accum_every):
                img = next(self.dl)
                img = img
                with autocast(enabled=self.amp):
                    loss = self.vae(img, return_discr_loss=True)
                    self.discr_scaler.scale(loss / self.grad_accum_every).backward()
                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})
            self.discr_scaler.step(self.discr_optim)
            self.discr_scaler.update()
            self.discr_optim.zero_grad()
            None
        self.ema_vae.update()
        if not steps % self.save_results_every:
            for model, filename in ((self.ema_vae.ema_model, f'{steps}.ema'), (self.vae, str(steps))):
                model.eval()
                imgs = next(self.dl)
                imgs = imgs
                recons = model(imgs)
                nrows = int(sqrt(self.batch_size))
                imgs_and_recons = torch.stack((imgs, recons), dim=0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0.0, 1.0)
                grid = make_grid(imgs_and_recons, nrow=2, normalize=True, value_range=(0, 1))
                logs['reconstructions'] = grid
                save_image(grid, str(self.results_folder / f'{filename}.png'))
            None
        if not steps % self.save_model_every:
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)
            ema_state_dict = self.ema_vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            torch.save(ema_state_dict, model_path)
            None
        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        device = next(self.vae.parameters()).device
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
        None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChanLayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEmbedLayer,
     lambda: ([], {'dim_in': 4, 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormChan,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwiGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleCombiner,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_DALLE2_pytorch(_paritybench_base):
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

