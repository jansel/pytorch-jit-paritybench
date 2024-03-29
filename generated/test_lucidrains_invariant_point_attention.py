import sys
_module = sys.modules[__name__]
del sys
denoise = _module
invariant_point_attention = _module
invariant_point_attention = _module
utils = _module
setup = _module
invariance = _module

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


import torch.nn.functional as F


from torch import nn


from torch import einsum


from torch.optim import Adam


from torch.cuda.amp import autocast


from torch import sin


from torch import cos


from torch import atan2


from torch import acos


from functools import wraps


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


class InvariantPointAttention(nn.Module):

    def __init__(self, *, dim, heads=8, scalar_key_dim=16, scalar_value_dim=16, point_key_dim=4, point_value_dim=4, pairwise_repr_dim=None, require_pairwise_repr=True, eps=1e-08):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr
        num_attn_logits = 3 if require_pairwise_repr else 2
        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5
        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias=False)
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = nn.Parameter(point_weight_init_value)
        self.point_attn_logits_scale = (num_attn_logits * point_key_dim * (9 / 2)) ** -0.5
        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias=False)
        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0
        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5
            self.to_pairwise_attn_bias = nn.Sequential(nn.Linear(pairwise_repr_dim, heads), Rearrange('b ... h -> (b h) ...'))
        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(self, single_repr, pairwise_repr=None, *, rotations, translations, mask=None):
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
        assert not (require_pairwise_repr and not exists(pairwise_repr)), 'pairwise representation must be given as second argument'
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)
        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h=h, c=3), (q_point, k_point, v_point))
        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations = repeat(translations, 'b n c -> (b h) n () c', h=h)
        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations
        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale
        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale
        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim=(-1, -2))
        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b=b)
        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)
        attn_logits = attn_logits_scalar + attn_logits_points
        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)
        attn = attn_logits.softmax(dim=-1)
        with disable_tf32(), autocast(enabled=False):
            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)
            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h=h)
            if require_pairwise_repr:
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)
            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)
            results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)
        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h=h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h=h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h=h)
        results = results_scalar, results_points, results_points_norm
        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h=h)
            results = *results, results_pairwise
        results = torch.cat(results, dim=-1)
        return self.to_out(results)


def FeedForward(dim, mult=1.0, num_layers=2, act=nn.ReLU):
    layers = []
    dim_hidden = dim * mult
    for ind in range(num_layers):
        is_first = ind == 0
        is_last = ind == num_layers - 1
        dim_in = dim if is_first else dim_hidden
        dim_out = dim if is_last else dim_hidden
        layers.append(nn.Linear(dim_in, dim_out))
        if is_last:
            continue
        layers.append(act())
    return nn.Sequential(*layers)


class IPABlock(nn.Module):

    def __init__(self, *, dim, ff_mult=1, ff_num_layers=3, post_norm=True, post_attn_dropout=0.0, post_ff_dropout=0.0, **kwargs):
        super().__init__()
        self.post_norm = post_norm
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(dim=dim, **kwargs)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm
        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x
        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        x = self.ff_norm(x) if post_norm else x
        return x


class IPATransformer(nn.Module):

    def __init__(self, *, dim, depth, num_tokens=None, predict_points=False, detach_rotations=True, **kwargs):
        super().__init__()
        try:
            self.quaternion_to_matrix = quaternion_to_matrix
            self.quaternion_multiply = quaternion_multiply
        except (ImportError, ModuleNotFoundError) as err:
            None
            raise err
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([IPABlock(dim=dim, **kwargs), nn.Linear(dim, 6)]))
        self.detach_rotations = detach_rotations
        self.predict_points = predict_points
        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(self, single_repr, *, translations=None, quaternions=None, pairwise_repr=None, mask=None):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape
        if exists(self.token_emb):
            x = self.token_emb(x)
        if not exists(quaternions):
            quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            quaternions = repeat(quaternions, 'd -> b n d', b=b, n=n)
        if not exists(translations):
            translations = torch.zeros((b, n, 3), device=device)
        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)
            if self.detach_rotations:
                rotations = rotations.detach()
            x = block(x, pairwise_repr=pairwise_repr, rotations=rotations, translations=translations)
            quaternion_update, translation_update = to_update(x).chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.0)
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)
            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)
        if not self.predict_points:
            return x, translations, quaternions
        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
        return points_global

