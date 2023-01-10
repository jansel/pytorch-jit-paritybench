import sys
_module = sys.modules[__name__]
del sys
point_transformer_pytorch = _module
multihead_point_transformer_pytorch = _module
point_transformer_pytorch = _module
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


import torch


from torch import nn


from torch import einsum


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


class MultiheadPointTransformerLayer(nn.Module):

    def __init__(self, *, dim, heads=4, dim_head=64, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.num_neighbors = num_neighbors
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(3, pos_mlp_hidden_dim), nn.ReLU(), nn.Linear(pos_mlp_hidden_dim, inner_dim))
        attn_inner_dim = inner_dim * attn_mlp_hidden_mult
        self.attn_mlp = nn.Sequential(nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=heads), nn.ReLU(), nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=heads))

    def forward(self, x, pos, mask=None):
        n, h, num_neighbors = x.shape[1], self.heads, self.num_neighbors
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        rel_pos = rearrange(pos, 'b i c -> b i 1 c') - rearrange(pos, 'b j c -> b 1 j c')
        rel_pos_emb = self.pos_mlp(rel_pos)
        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)
        qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - rearrange(k, 'b h j d -> b h 1 j d')
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')
        v = repeat(v, 'b h j d -> b h i j d', i=n)
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)
            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)
            dist, indices = rel_dist.topk(num_neighbors, largest=False)
            indices_with_heads = repeat(indices, 'b i j -> b h i j', h=h)
            v = batched_index_select(v, indices_with_heads, dim=3)
            qk_rel = batched_index_select(qk_rel, indices_with_heads, dim=3)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices_with_heads, dim=3)
            if exists(mask):
                mask = batched_index_select(mask, indices, dim=2)
        v = v + rel_pos_emb
        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')
        sim = self.attn_mlp(attn_mlp_input)
        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)
        attn = sim.softmax(dim=-2)
        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = einsum('b d i j, b i j d -> b i d', attn, v)
        return self.to_out(agg)


class PointTransformerLayer(nn.Module):

    def __init__(self, *, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.pos_mlp = nn.Sequential(nn.Linear(3, pos_mlp_hidden_dim), nn.ReLU(), nn.Linear(pos_mlp_hidden_dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim * attn_mlp_hidden_mult), nn.ReLU(), nn.Linear(dim * attn_mlp_hidden_mult, dim))

    def forward(self, x, pos, mask=None):
        n, num_neighbors = x.shape[1], self.num_neighbors
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)
        qk_rel = q[:, :, None, :] - k[:, None, :, :]
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]
        v = repeat(v, 'b j d -> b i j d', i=n)
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)
            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)
            dist, indices = rel_dist.topk(num_neighbors, largest=False)
            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)
            mask = batched_index_select(mask, indices, dim=2) if exists(mask) else None
        v = v + rel_pos_emb
        sim = self.attn_mlp(qk_rel + rel_pos_emb)
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)
        attn = sim.softmax(dim=-2)
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg

