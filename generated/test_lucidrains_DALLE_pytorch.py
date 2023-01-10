import sys
_module = sys.modules[__name__]
del sys
dalle_pytorch = _module
attention = _module
dalle_pytorch = _module
distributed_backends = _module
deepspeed_backend = _module
distributed_backend = _module
dummy_backend = _module
horovod_backend = _module
distributed_utils = _module
loader = _module
reversible = _module
tokenizer = _module
transformer = _module
vae = _module
version = _module
generate = _module
setup = _module
train_dalle = _module
train_vae = _module

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


from inspect import isfunction


from math import ceil


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


from math import log2


from math import sqrt


import numpy as np


from random import randint


from random import choice


from torch.utils.data import Dataset


from torchvision import transforms as T


import torch.nn as nn


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


from functools import lru_cache


from collections import deque


from collections.abc import Iterable


from functools import partial


from itertools import islice


from itertools import cycle


import warnings


from math import log


from torchvision.utils import make_grid


from torchvision.utils import save_image


import time


from torch.nn.utils import clip_grad_norm_


from torch.optim import Adam


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import DataLoader


import math


from torch.optim.lr_scheduler import ExponentialLR


from torchvision.datasets import ImageFolder


def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))


def exists(val):
    return val is not None


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def stable_softmax(t, dim=-1, alpha=32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


class Attention(nn.Module):

    def __init__(self, dim, seq_len, causal=True, heads=8, dim_head=64, dropout=0.0, stable=False, static_mask=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5
        self.stable = stable
        self.causal = causal
        self.register_buffer('static_mask', static_mask, persistent=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, rotary_pos_emb=None, cache=None, cache_key=None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax
        offset = cache.get('offset', 0) if exists(cache) else 0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb[..., offset:, :], (q, k, v))
        q = q * self.scale
        if offset > 0:
            k_top, v_top = cache[cache_key]
            k = torch.cat([k_top, k], dim=-2)
            v = torch.cat([v_top, v], dim=-2)
        if exists(cache):
            cache[cache_key] = k, v
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        if self.causal and offset == 0:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)
        if exists(self.static_mask):
            dots.masked_fill_(~self.static_mask[offset:offset + n, :offset + n], mask_value)
        attn = softmax(dots, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


def default(val, d):
    return val if exists(val) else d


class SparseConvCausalAttention(nn.Module):

    def __init__(self, dim, seq_len, image_size=32, kernel_size=5, dilation=1, heads=8, dim_head=64, dropout=0.0, stable=False, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stable = stable
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, rotary_pos_emb=None):
        b, n, _, h, img_size, kernel_size, dilation, seq_len, device = *x.shape, self.heads, self.image_size, self.kernel_size, self.dilation, self.seq_len, x.device
        softmax = torch.softmax if not self.stable else stable_softmax
        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len
        padding = seq_len - n + 1
        mask = default(mask, lambda : torch.ones(b, text_len, device=device).bool())
        x = F.pad(x, (0, 0, 0, padding), value=0)
        mask = mask[:, :text_len]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), qkv)
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        q *= self.scale
        (q_text, q_img), (k_text, k_img), (v_text, v_img) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))
        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)
        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)
        attn_text = softmax(dots_text, dim=-1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        same_padding = effective_kernel_size // 2
        causal_padding = same_padding * 2, 0, same_padding * 2, 0
        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h=img_size), (k_img, v_img))
        k_img, v_img = map(lambda t: F.pad(t, causal_padding), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, dilation=dilation), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j=kernel_size ** 2), (k_img, v_img))
        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
        dots_image_to_text = einsum('b i d, b j d -> b i j', q_img, k_text)
        i, j = dots_image.shape[-2:]
        ones = torch.ones((img_seq_len,), device=device)
        ones = rearrange(ones, '(h w) -> () () h w', h=img_size)
        ones = F.pad(ones, causal_padding, value=0.0)
        ones = F.unfold(ones, kernel_size, dilation=dilation)
        ones = rearrange(ones, 'b j i -> b i j')
        padding_mask = ones == 0.0
        padding_mask = repeat(padding_mask, '() i j -> b i j', b=b * h)
        mask = repeat(mask, 'b j -> (b h) i j', i=i, h=h)
        mask = torch.cat((~mask, padding_mask), dim=-1)
        dots = torch.cat((dots_image_to_text, dots_image), dim=-1)
        dots.masked_fill_(mask, mask_value)
        attn = softmax(dots, dim=-1)
        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]
        out_image_to_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)
        out_image_to_text = einsum('b i j, b j d -> b i d', attn_image_to_text, v_text)
        out_image = out_image_to_image + out_image_to_text
        out = torch.cat((out_text, out_image), dim=1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out[:, :n]


class SparseAxialCausalAttention(nn.Module):

    def __init__(self, dim, seq_len, image_size=32, axis=0, heads=8, dim_head=64, dropout=0.0, stable=False, **kwargs):
        super().__init__()
        assert axis in {0, 1}, 'axis must be either 0 (along height) or 1 (along width)'
        self.axis = axis
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.stable = stable
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, rotary_pos_emb=None):
        b, n, _, h, img_size, axis, seq_len, device = *x.shape, self.heads, self.image_size, self.axis, self.seq_len, x.device
        softmax = torch.softmax if not self.stable else stable_softmax
        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len
        padding = seq_len - n + 1
        mask = default(mask, lambda : torch.ones(b, text_len, device=device).bool())
        x = F.pad(x, (0, 0, 0, padding), value=0)
        mask = mask[:, :text_len]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), qkv)
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        q *= self.scale
        (q_text, q_img), (k_text, k_img), (v_text, v_img) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))
        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)
        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)
        attn_text = softmax(dots_text, dim=-1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)
        split_axis_einops = 'b (h w) c -> b h w c' if axis == 0 else 'b (h w) c -> b w h c'
        merge_axis_einops = 'b x n d -> b (x n) d' if axis == 0 else 'b x n d -> b (n x) d'
        q_img, k_img, v_img = map(lambda t: rearrange(t, split_axis_einops, h=img_size), (q_img, k_img, v_img))
        dots_image_to_image = einsum('b x i d, b x j d -> b x i j', q_img, k_img)
        dots_image_to_text = einsum('b x i d, b j d -> b x i j', q_img, k_text)
        dots = torch.cat((dots_image_to_text, dots_image_to_image), dim=-1)
        bh, x, i, j = dots.shape
        causal_mask = torch.ones(i, img_size, device=device).triu_(img_size - i + 1).bool()
        causal_mask = repeat(causal_mask, 'i j -> b x i j', b=bh, x=x)
        mask = repeat(mask, 'b j -> (b h) x i j', h=h, x=x, i=i)
        mask = torch.cat((~mask, causal_mask), dim=-1)
        dots.masked_fill_(mask, mask_value)
        attn = softmax(dots, dim=-1)
        attn_image_to_text, attn_image_to_image = attn[..., :text_len], attn[..., text_len:]
        out_image_to_image = einsum('b x i j, b x j d -> b x i d', attn_image_to_image, v_img)
        out_image_to_text = einsum('b x i j, b j d -> b x i d', attn_image_to_text, v_text)
        out_image = out_image_to_image + out_image_to_text
        out_image = rearrange(out_image, merge_axis_einops, x=img_size)
        out = torch.cat((out_text, out_image), dim=1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out[:, :n]


class SparseAttention(Attention):

    def __init__(self, *args, block_size=16, text_seq_len=256, num_random_blocks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)
        global_block_indices = list(range(ceil(text_seq_len / block_size)))
        self.attn_fn = SparseSelfAttention(sparsity_config=VariableSparsityConfig(num_heads=self.heads, block=self.block_size, num_random_blocks=num_random_blocks, global_block_indices=global_block_indices, attention='unidirectional' if self.causal else 'bidirectional'), max_seq_length=self.seq_len, attn_mask_mode='add')

    def forward(self, x, mask=None, rotary_pos_emb=None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        remainder = n % self.block_size
        mask = default(mask, lambda : torch.ones(b, n, device=device).bool())
        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value=0)
            mask = F.pad(mask, (0, padding), value=False)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        key_pad_mask = None
        if exists(mask):
            key_pad_mask = ~mask
        attn_mask = None
        if self.causal:
            i, j = q.shape[-2], k.shape[-2]
            mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            attn_mask = torch.zeros(i, j, device=device)
            mask_value = max_neg_value(q) / 2
            attn_mask.masked_fill_(mask, mask_value)
        out = self.attn_fn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out[:, :n]


class SharedEmbedding(nn.Embedding):

    def __init__(self, linear, start_index, end_index, **kwargs):
        super().__init__(end_index - start_index, linear.weight.shape[1], **kwargs)
        del self.weight
        self.linear = linear
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, input):
        return F.embedding(input, self.linear.weight[self.start_index:self.end_index], self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class ResBlock(nn.Module):

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan, 3, padding=1), nn.ReLU(), nn.Conv2d(chan, chan, 3, padding=1), nn.ReLU(), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class DiscreteVAE(nn.Module):

    def __init__(self, image_size=256, num_tokens=512, codebook_dim=512, num_layers=3, num_resnet_blocks=0, hidden_dim=64, channels=3, smooth_l1_loss=False, temperature=0.9, straight_through=False, kl_div_loss_weight=0.0, normalization=((*((0.5,) * 3), 0), (*((0.5,) * 3), 1))):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0
        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        hdim = hidden_dim
        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))
        enc_chans = [channels, *enc_chans]
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))
        enc_layers = []
        dec_layers = []
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))
        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))
        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight
        self.normalization = tuple(map(lambda t: t[:channels], normalization))
        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if not distributed_utils.is_distributed or not distributed_utils.using_backend(distributed_utils.DeepSpeedBackend):
            return
        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images
        means, stds = map(lambda t: torch.as_tensor(t), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(self, img, return_loss=False, return_recons=False, return_logits=False, temp=None):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'
        img = self.norm(img)
        logits = self.encoder(img)
        if return_logits:
            return logits
        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)
        if not return_loss:
            return out
        recon_loss = self.loss_fn(img, out)
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1.0 / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        loss = recon_loss + kl_div * kl_div_loss_weight
        if not return_recons:
            return loss
        return loss, out


class CachedAs(nn.Module):
    """
    A wrapper that defines a key for the inference cache.
    """

    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, dropout=0.0, mult=4.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x, cache=None, cache_key=None):
        return self.net(x)


class LayerScale(nn.Module):

    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-05
        else:
            init_eps = 1e-06
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class NonCached(nn.Module):
    """
    A wrapper for layers that don't support the inference cache themselves.
    Reconstructs the full sequence before the layer and
    cuts the suffix of the outputs after the layer.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *, cache=None, cache_key=None, **kwargs):
        n = x.shape[-2]
        if exists(cache):
            if cache_key in cache:
                x = torch.cat([cache[cache_key], x], dim=-2)
            cache[cache_key] = x
        out = self.fn(x, **kwargs)
        return out[:, -n:]


class PreNorm(nn.Module):

    def __init__(self, dim, fn, sandwich=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


class PreShiftToken(nn.Module):

    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1

    def forward(self, x, cache=None, cache_key=None, **kwargs):
        seq_len, image_size, text_len = self.seq_len, self.image_size, self.text_len
        if exists(cache) and cache_key in cache:
            offset = cache['offset']
            assert offset >= text_len, 'cached inference for text is not supported'
            q = cache[cache_key]
            assert isinstance(q, deque) and len(q) == image_size
            x_top, x_left, *x_pass = x[:, -1].chunk(4, dim=-1)
            q.append((x_top, x_left))
            x_top = q.popleft()[0]
            x_left = q[-2][1]
            if (offset - text_len) % image_size == 0:
                x_left = torch.zeros_like(x_left)
            x = torch.cat((x_top, x_left, *x_pass), dim=-1)
            return self.fn(x[:, None], cache=cache, **kwargs)
        n = x.shape[1]
        padding = seq_len - n + 1
        if n < text_len:
            return self.fn(x, **kwargs)
        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h=image_size)
        x_text_shift, x_text_pass = x_text.chunk(2, dim=-1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim=-1)
        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim=-1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim=-1)
        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x_img = x_img[:, :-padding]
        x = torch.cat((x_text, x_img), dim=1)
        if exists(cache):
            dummy_top, dummy_left, *_ = x[:, -1].chunk(4, dim=-1)
            dummy_top, dummy_left = torch.zeros_like(dummy_top), torch.zeros_like(dummy_left)
            q = deque()
            x_img = x_img[:, -image_size:]
            for _ in range(image_size - x_img.shape[1]):
                q.append((dummy_top, dummy_left))
            for i in range(x_img.shape[1]):
                q.append(x_img[:, i].chunk(4, dim=-1)[:2])
            cache[cache_key] = q
        return self.fn(x, cache=cache, **kwargs)


class Deterministic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


class _ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]
    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: {key: val} if route else {}, routes)
            routed_args[depth] = {**f_args, **new_f_args}, {**g_args, **new_g_args}
    return routed_args


class ReversibleSequence(nn.Module):

    def __init__(self, blocks, args_route={}):
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))
        out = _ReversibleFunction.apply(x, blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).mean(dim=0)


class SequentialSequence(nn.Module):

    def __init__(self, layers, args_route={}, layer_dropout=0.0):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))
        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


def cast_tuple(val, depth=1):
    return val if isinstance(val, Iterable) else (val,) * depth


class Transformer(nn.Module):

    def __init__(self, *, dim, depth, seq_len, reversible=False, causal=True, heads=8, dim_head=64, ff_mult=4, attn_dropout=0.0, ff_dropout=0.0, attn_types=None, image_fmap_size=None, sparse_attn=False, stable=False, sandwich_norm=False, shift_tokens=False, rotary_emb=True, shared_attn_ids=None, shared_ff_ids=None, optimize_for_inference=False):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)
        self.seq_len = seq_len
        self.image_fmap_size = image_fmap_size
        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)
        shared_attn_ids = cycle(default(shared_attn_ids, range(depth)))
        shared_ff_ids = cycle(default(shared_ff_ids, range(depth)))
        shared_attn_layers = {}
        shared_ff_layers = {}
        for ind, sparse_attn, attn_type, attn_id, ff_id in zip(range(depth), sparse_layer, attn_type_layer, shared_attn_ids, shared_ff_ids):
            if attn_type == 'full':
                attn_class = partial(Attention, stable=stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                if optimize_for_inference:
                    attn_class = partial(Attention, stable=stable, static_mask=self._get_attention_mask(attn_type))
                else:
                    attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=0, image_size=image_fmap_size, stable=stable)
            elif attn_type == 'axial_col':
                if optimize_for_inference:
                    attn_class = partial(Attention, stable=stable, static_mask=self._get_attention_mask(attn_type))
                else:
                    attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=1, image_size=image_fmap_size, stable=stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len=seq_len, image_size=image_fmap_size, stable=stable)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')
            attn, reused_attn_type = shared_attn_layers.get(attn_id, (None, None))
            if not exists(attn):
                attn = attn_class(dim, causal=causal, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout)
                shared_attn_layers[attn_id] = attn, attn_type
            elif attn_type != reused_attn_type:
                raise ValueError(f'attn_types do not match shared_attn_ids (ind = {ind}, attn_type = "{attn_type}", reused_attn_type = "{reused_attn_type}")')
            ff = shared_ff_layers.get(ff_id)
            if not exists(ff):
                ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
                shared_ff_layers[ff_id] = ff
            if isinstance(attn, Attention):
                attn = CachedAs(f'attn_{ind}', attn)
            else:
                attn = NonCached(attn)
            if shift_tokens:
                attn = CachedAs(f'preshift_attn_{ind}', PreShiftToken(attn, image_size=image_fmap_size, seq_len=seq_len))
                ff = CachedAs(f'preshift_ff_{ind}', PreShiftToken(ff, image_size=image_fmap_size, seq_len=seq_len))
            layers.append(nn.ModuleList([LayerScale(dim, ind + 1, PreNorm(dim, attn, sandwich=sandwich_norm)), LayerScale(dim, ind + 1, PreNorm(dim, ff, sandwich=sandwich_norm))]))
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        route_all = ((True, True),) * depth
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn, 'cache': route_all}
        self.layers = execute_type(layers, args_route=attn_route_map)
        pos_emb = None
        if rotary_emb:
            rot_dim = dim_head // 3
            img_seq_len = image_fmap_size ** 2
            text_len = seq_len - img_seq_len + 1
            text_pos_emb = RotaryEmbedding(dim=rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim=rot_dim, freqs_for='pixel')
            text_freqs = text_pos_emb(torch.arange(text_len))
            img_to_text_freqs = text_pos_emb(torch.full((img_seq_len,), 8192))
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim=0)
            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps=image_fmap_size))
            img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'), rearrange(img_freqs_axial, 'j d -> () j d')), dim=-1)
            img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')
            text_axial_freqs = img_axial_pos_emb(torch.full((text_len,), -10.0))
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim=-1)
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim=0)
            pos_emb = torch.cat((text_freqs, img_freqs), dim=-1)
            pos_emb = rearrange(pos_emb, 'n d -> () n d')
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb=self.pos_emb, **kwargs)

    def _get_attention_mask(self, attn_type):
        img_seq_len = self.image_fmap_size ** 2
        text_len = self.seq_len + 1 - img_seq_len
        static_mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        static_mask[:, :text_len] = True
        if attn_type == 'axial_row':
            for row in range(self.image_fmap_size):
                begin = text_len + row * self.image_fmap_size
                end = text_len + (row + 1) * self.image_fmap_size
                static_mask[begin:end, begin:end] = True
        elif attn_type == 'axial_col':
            for col in range(self.image_fmap_size):
                begin = text_len + col
                static_mask[begin::self.image_fmap_size, begin::self.image_fmap_size] = True
        else:
            raise ValueError(f'attention type "{attn_type}" can\'t be simulated with a static mask')
        return static_mask


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


class CLIP(nn.Module):

    def __init__(self, *, dim_text=512, dim_image=512, dim_latent=512, num_text_tokens=10000, text_enc_depth=6, text_seq_len=256, text_heads=8, num_visual_tokens=512, visual_enc_depth=6, visual_heads=8, visual_image_size=256, visual_patch_size=32, channels=3):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth, heads=text_heads, rotary_emb=False)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)
        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2
        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        self.visual_transformer = Transformer(causal=False, seq_len=num_patches, dim=dim_image, depth=visual_enc_depth, heads=visual_heads, rotary_emb=False)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, text, image, text_mask=None, return_loss=False):
        b, device, p = text.shape[0], text.device, self.visual_patch_size
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device=device))
        enc_text = self.text_transformer(text_emb, mask=text_mask)
        enc_image = self.visual_transformer(image_emb)
        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim=1)
        else:
            text_latents = enc_text.mean(dim=1)
        image_latents = enc_image.mean(dim=1)
        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)
        text_latents, image_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, image_latents))
        temp = self.temperature.exp()
        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim
        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


class DivideMax(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / maxes


OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'


OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'


def get_pkg_version(pkg_name):
    return get_distribution(pkg_name).version


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


class OpenAIDiscreteVAE(nn.Module):

    def __init__(self):
        super().__init__()
        assert version.parse(get_pkg_version('torch')) < version.parse('1.11.0'), 'torch version must be <= 1.10 in order to use OpenAI discrete vae'
        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))
        make_contiguous(self)
        self.channels = 3
        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim=1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h=int(sqrt(n)))
        z = F.one_hot(img_seq, num_classes=self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented


VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'


VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class VQGanVAE(nn.Module):

    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()
        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config['model'])
        state = torch.load(model_path, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=False)
        None
        self.model = model
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(f) / log(2))
        self.channels = 3
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)
        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if not distributed_utils.is_distributed or not distributed_utils.using_backend(distributed_utils.DeepSpeedBackend):
            return
        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.model.quantize.embed.weight if self.is_gumbel else self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = 2 * img - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel else one_hot_indices @ self.model.quantize.embedding.weight
        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)
        img = (img.clamp(-1.0, 1.0) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented


class always:

    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / temperature + gumbel_noise(t)).argmax(dim=dim)


def is_empty(t):
    return t.nelement() == 0


def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class DALLE(nn.Module):

    def __init__(self, *, dim, vae, num_text_tokens=10000, text_seq_len=256, depth, heads=8, dim_head=64, reversible=False, attn_dropout=0.0, ff_dropout=0, sparse_attn=False, attn_types=None, loss_img_weight=7, stable=False, sandwich_norm=False, shift_tokens=True, rotary_emb=True, shared_attn_ids=None, shared_ff_ids=None, share_input_output_emb=False, optimize_for_inference=False):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.image_size // 2 ** vae.num_layers
        image_seq_len = image_fmap_size ** 2
        num_text_tokens = num_text_tokens + text_seq_len
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0)
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(image_fmap_size, image_fmap_size)) if not rotary_emb else always(0)
        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len
        self.vae = vae
        set_requires_grad(self.vae, False)
        self.transformer = Transformer(dim=dim, causal=True, seq_len=seq_len, depth=depth, heads=heads, dim_head=dim_head, reversible=reversible, attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_types=attn_types, image_fmap_size=image_fmap_size, sparse_attn=sparse_attn, stable=stable, sandwich_norm=sandwich_norm, shift_tokens=shift_tokens, rotary_emb=rotary_emb, shared_attn_ids=shared_attn_ids, shared_ff_ids=shared_ff_ids, optimize_for_inference=optimize_for_inference)
        self.stable = stable
        if stable:
            self.norm_by_max = DivideMax(dim=-1)
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.total_tokens))
        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits[1], 0, num_text_tokens)
            self.image_emb = SharedEmbedding(self.to_logits[1], num_text_tokens, total_tokens)
        else:
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.image_emb = nn.Embedding(num_image_tokens, dim)
        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)
        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')
        logits_mask = (seq_range >= text_seq_len) & (logits_range < num_text_tokens) | (seq_range < text_seq_len) & (logits_range >= num_text_tokens)
        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight

    @torch.no_grad()
    @eval_decorator
    def generate_texts(self, tokenizer, text=None, *, filter_thres=0.5, temperature=1.0):
        text_seq_len = self.text_seq_len
        if text is None or text == '':
            text_tokens = torch.tensor([[0]])
        else:
            text_tokens = torch.tensor(tokenizer.tokenizer.encode(text)).unsqueeze(0)
        for _ in range(text_tokens.shape[1], text_seq_len):
            device = text_tokens.device
            tokens = self.text_emb(text_tokens)
            tokens += self.text_pos_emb(torch.arange(text_tokens.shape[1], device=device))
            seq_len = tokens.shape[1]
            output_transf = self.transformer(tokens)
            if self.stable:
                output_transf = self.norm_by_max(output_transf)
            logits = self.to_logits(output_transf)
            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            text_tokens = torch.cat((text_tokens, sample[:, None]), dim=-1)
        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts

    @torch.no_grad()
    @eval_decorator
    def generate_images(self, text, *, clip=None, filter_thres=0.5, temperature=1.0, img=None, num_init_img_tokens=None, cond_scale=1.0, use_cache=False):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len
        text = text[:, :text_seq_len]
        out = text
        if exists(img):
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'
            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'
            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)
        prev_cache = None
        cache = {} if use_cache else None
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len
            text, image = out[:, :text_seq_len], out[:, text_seq_len:]
            logits = self.forward_with_cond_scale(text, image, cond_scale=cond_scale, cache=cache)
            logits = logits[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            sample -= num_text_tokens if is_image else 0
            out = torch.cat((out, sample[:, None]), dim=-1)
        text_seq = out[:, :text_seq_len]
        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)
        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores
        return images

    def forward_with_cond_scale(self, *args, cond_scale=1, cache=None, **kwargs):
        if cond_scale == 1:
            return self(*args, **kwargs)
        prev_cache = cache.copy() if exists(cache) else None
        logits = self(*args, cache=cache, **kwargs)
        null_cond_logits = self(*args, null_cond_prob=1.0, cache=prev_cache, **kwargs)
        return null_cond_logits + (logits - null_cond_logits) * cond_scale

    def forward(self, text, image=None, return_loss=False, null_cond_prob=0.0, cache=None):
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len
        if null_cond_prob > 0:
            null_mask = prob_mask_like((batch,), null_cond_prob, device=device)
            text *= rearrange(~null_mask, 'b -> b 1')
        text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)
        text = F.pad(text, (1, 0), value=0)
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))
        seq_len = tokens.shape[1]
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image_size = self.vae.image_size
                channels = self.vae.channels
                assert tuple(image.shape[1:]) == (channels, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
                image = self.vae.get_codebook_indices(image)
            image_len = image.shape[1]
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            tokens = torch.cat((tokens, image_emb), dim=1)
            seq_len += image_len
        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]
        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)
        if exists(cache) and cache.get('offset'):
            tokens = tokens[:, -1:]
        out = self.transformer(tokens, cache=cache)
        if self.stable:
            out = self.norm_by_max(out)
        logits = self.to_logits(out)
        logits_mask = self.logits_mask[:, :seq_len]
        if exists(cache) and cache.get('offset'):
            logits_mask = logits_mask[:, -1:]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)
        if exists(cache):
            cache['offset'] = cache.get('offset', 0) + logits.shape[1]
        if not return_loss:
            return logits
        assert exists(image), 'when training, image must be supplied'
        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim=1)
        logits = rearrange(logits, 'b n c -> b c n')
        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])
        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Deterministic,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DivideMax,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale,
     lambda: ([], {'dim': 4, 'depth': 1, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NonCached,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lucidrains_DALLE_pytorch(_paritybench_base):
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

