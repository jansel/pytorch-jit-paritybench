import sys
_module = sys.modules[__name__]
del sys
crossvit = _module
module = _module

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


import torch.nn.functional as F


import numpy as np


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads
        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim=96, small_depth=4, small_heads=3, small_dim_head=32, small_mlp_dim=384, large_dim=192, large_depth=1, large_heads=3, large_dim_head=64, large_mlp_dim=768, cross_attn_depth=1, cross_attn_heads=3, dropout=0.0):
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([nn.Linear(small_dim, large_dim), nn.Linear(large_dim, small_dim), PreNorm(large_dim, CrossAttention(large_dim, heads=cross_attn_heads, dim_head=large_dim_head, dropout=dropout)), nn.Linear(large_dim, small_dim), nn.Linear(small_dim, large_dim), PreNorm(small_dim, CrossAttention(small_dim, heads=cross_attn_heads, dim_head=small_dim_head, dropout=dropout))]))

    def forward(self, xs, xl):
        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)
        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)
        return xs, xl


class CrossViT(nn.Module):

    def __init__(self, image_size, channels, num_classes, patch_size_small=14, patch_size_large=16, small_dim=96, large_dim=192, small_depth=1, large_depth=4, cross_attn_depth=1, multi_scale_enc_depth=3, heads=3, pool='cls', dropout=0.0, emb_dropout=0.0, scale_dim=4):
        super().__init__()
        assert image_size % patch_size_small == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_small = (image_size // patch_size_small) ** 2
        patch_dim_small = channels * patch_size_small ** 2
        assert image_size % patch_size_large == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_large = (image_size // patch_size_large) ** 2
        patch_dim_large = channels * patch_size_large ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding_small = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_small, p2=patch_size_small), nn.Linear(patch_dim_small, small_dim))
        self.to_patch_embedding_large = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_large, p2=patch_size_large), nn.Linear(patch_dim_large, large_dim))
        self.pos_embedding_small = nn.Parameter(torch.randn(1, num_patches_small + 1, small_dim))
        self.cls_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.dropout_small = nn.Dropout(emb_dropout)
        self.pos_embedding_large = nn.Parameter(torch.randn(1, num_patches_large + 1, large_dim))
        self.cls_token_large = nn.Parameter(torch.randn(1, 1, large_dim))
        self.dropout_large = nn.Dropout(emb_dropout)
        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim, small_depth=small_depth, small_heads=heads, small_dim_head=small_dim // heads, small_mlp_dim=small_dim * scale_dim, large_dim=large_dim, large_depth=large_depth, large_heads=heads, large_dim_head=large_dim // heads, large_mlp_dim=large_dim * scale_dim, cross_attn_depth=cross_attn_depth, cross_attn_heads=heads, dropout=dropout))
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head_small = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, num_classes))
        self.mlp_head_large = nn.Sequential(nn.LayerNorm(large_dim), nn.Linear(large_dim, num_classes))

    def forward(self, img):
        xs = self.to_patch_embedding_small(img)
        b, n, _ = xs.shape
        cls_token_small = repeat(self.cls_token_small, '() n d -> b n d', b=b)
        xs = torch.cat((cls_token_small, xs), dim=1)
        xs += self.pos_embedding_small[:, :n + 1]
        xs = self.dropout_small(xs)
        xl = self.to_patch_embedding_large(img)
        b, n, _ = xl.shape
        cls_token_large = repeat(self.cls_token_large, '() n d -> b n d', b=b)
        xl = torch.cat((cls_token_large, xl), dim=1)
        xl += self.pos_embedding_large[:, :n + 1]
        xl = self.dropout_large(xl)
        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xl = multi_scale_transformer(xs, xl)
        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
        xl = xl.mean(dim=1) if self.pool == 'mean' else xl[:, 0]
        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)
        x = xs + xl
        return x


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_rishikksh20_CrossViT_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

