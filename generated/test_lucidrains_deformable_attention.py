import sys
_module = sys.modules[__name__]
del sys
deformable_attention = _module
deformable_attention_1d = _module
deformable_attention_2d = _module
deformable_attention_3d = _module
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


import torch.nn.functional as F


from torch import nn


from torch import einsum


class Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * rearrange(self.scale, 'c -> 1 c 1 1 1')


class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups
        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(nn.Linear(3, dim), nn.ReLU()))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU()))
        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype
        grid_q = rearrange(grid_q, '... c -> 1 (...) c')
        grid_kv = rearrange(grid_kv, 'b ... c -> b (...) c')
        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)
        for layer in self.mlp:
            bias = layer(bias)
        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g=self.offset_groups)
        return bias


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return numer % denom == 0


def grid_sample_1d(feats, grid, *args, **kwargs):
    grid = rearrange(grid, '... -> ... 1 1')
    grid = F.pad(grid, (0, 1), value=0.0)
    feats = rearrange(feats, '... -> ... 1')
    out = F.grid_sample(feats, grid, **kwargs)
    return rearrange(out, '... 1 -> ...')


def normalize_grid(grid, dim=1, out_dim=-1):
    f, h, w = grid.shape[-3:]
    grid_f, grid_h, grid_w = grid.unbind(dim=dim)
    grid_f = 2.0 * grid_f / max(f - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
    return torch.stack((grid_f, grid_h, grid_w), dim=out_dim)


class DeformableAttention1D(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, downsample_factor=4, offset_scale=None, offset_groups=None, offset_kernel_size=6, cpb_log_distance=True, group_queries=True, group_key_values=True):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)
        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups
        offset_dims = inner_dim // offset_groups
        self.downsample_factor = downsample_factor
        self.to_offsets = nn.Sequential(nn.Conv1d(offset_dims, offset_dims, offset_kernel_size, groups=offset_dims, stride=downsample_factor, padding=(offset_kernel_size - downsample_factor) // 2), nn.GELU(), nn.Conv1d(offset_dims, 1, 1, bias=False), Rearrange('b 1 n -> b n'), nn.Tanh(), Scale(offset_scale))
        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2, log_distance=cpb_log_distance)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv1d(dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv1d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid=False):
        """
        b - batch
        h - heads
        n - sequence dimension
        d - dimension
        g - offset groups
        """
        heads, b, n, downsample_factor, device = self.heads, x.shape[0], x.shape[-1], self.downsample_factor, x.device
        q = self.to_q(x)
        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g=self.offset_groups)
        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)
        grid = torch.arange(offsets.shape[-1], device=device)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)
        kv_feats = grid_sample_1d(group(x), vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
        kv_feats = rearrange(kv_feats, '(b g) d n -> b (g d) n', b=b)
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        q = q * self.scale
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=heads), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        seq_range = torch.arange(n, device=device)
        seq_scaled = normalize_grid(seq_range, dim=0)
        rel_pos_bias = self.rel_pos_bias(seq_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        if return_vgrid:
            return out, vgrid
        return out


def create_grid_like(t, dim=0):
    f, h, w, device = *t.shape[-3:], t.device
    grid = torch.stack(torch.meshgrid(torch.arange(f, device=device), torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'), dim=dim)
    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid


class DeformableAttention2D(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, downsample_factor=4, offset_scale=None, offset_groups=None, offset_kernel_size=6, group_queries=True, group_key_values=True):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)
        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups
        offset_dims = inner_dim // offset_groups
        self.downsample_factor = downsample_factor
        self.to_offsets = nn.Sequential(nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups=offset_dims, stride=downsample_factor, padding=(offset_kernel_size - downsample_factor) // 2), nn.GELU(), nn.Conv2d(offset_dims, 2, 1, bias=False), nn.Tanh(), Scale(offset_scale))
        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid=False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """
        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device
        q = self.to_q(x)
        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g=self.offset_groups)
        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)
        grid = create_grid_like(offsets)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)
        kv_feats = F.grid_sample(group(x), vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b=b)
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        q = q * self.scale
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim=0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.to_out(out)
        if return_vgrid:
            return out, vgrid
        return out


def cast_tuple(x, length=1):
    return x if isinstance(x, tuple) else (x,) * depth


class DeformableAttention3D(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, downsample_factor=4, offset_scale=None, offset_groups=None, offset_kernel_size=6, group_queries=True, group_key_values=True):
        super().__init__()
        downsample_factor = cast_tuple(downsample_factor, length=3)
        offset_scale = default(offset_scale, downsample_factor)
        offset_conv_padding = tuple(map(lambda x: (x[0] - x[1]) / 2, zip(offset_kernel_size, downsample_factor)))
        assert all([(padding > 0 and padding.is_integer()) for padding in offset_conv_padding])
        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups
        offset_dims = inner_dim // offset_groups
        self.downsample_factor = downsample_factor
        self.to_offsets = nn.Sequential(nn.Conv3d(offset_dims, offset_dims, offset_kernel_size, groups=offset_dims, stride=downsample_factor, padding=tuple(map(int, offset_conv_padding))), nn.GELU(), nn.Conv3d(offset_dims, 3, 1, bias=False), nn.Tanh(), Scale(offset_scale))
        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv3d(dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv3d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_v = nn.Conv3d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid=False):
        """
        b - batch
        h - heads
        f - frames
        x - height
        y - width
        d - dimension
        g - offset groups
        """
        heads, b, f, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-3:], self.downsample_factor, x.device
        q = self.to_q(x)
        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g=self.offset_groups)
        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)
        grid = create_grid_like(offsets)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)
        kv_feats = F.grid_sample(group(x), vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b=b)
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        q = q * self.scale
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim=0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (f x y) d -> b (h d) f x y', f=f, x=h, y=w)
        out = self.to_out(out)
        if return_vgrid:
            return out, vgrid
        return out

