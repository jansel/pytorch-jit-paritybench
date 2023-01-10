import sys
_module = sys.modules[__name__]
del sys
robotic_transformer_pytorch = _module
robotic_transformer_pytorch = _module
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


from typing import List


from typing import Optional


from typing import Callable


from typing import Tuple


from functools import partial


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


def exists(val):
    return val is not None


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, cond_fn=None):
        x = self.norm(x)
        if exists(cond_fn):
            x = cond_fn(x)
        return self.net(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        self.gate = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(dim, hidden_dim, bias=False), nn.SiLU(), nn.Linear(hidden_dim, dim, bias=False), nn.Sigmoid(), Rearrange('b c -> b c 1 1'))

    def forward(self, x):
        return x * self.gate(x)


class Dropsample(nn.Module):

    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device
        if self.prob == 0.0 or not self.training:
            return x
        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):

    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Attention(nn.Module):

    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7):
        super().__init__()
        assert dim % dim_head == 0, 'dimension should be divisible by dimension per head'
        self.norm = LayerNorm(dim)
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads
        x = self.norm(x)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


def MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1
    net = nn.Sequential(nn.Conv2d(dim_in, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim), nn.BatchNorm2d(hidden_dim), nn.GELU(), SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate), nn.Conv2d(hidden_dim, dim_out, 1), nn.BatchNorm2d(dim_out))
    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)
    return net


def default(val, d):
    return val if exists(val) else d


class TransformerAttention(nn.Module):

    def __init__(self, dim, causal=False, dim_head=64, dim_context=None, heads=8, norm_context=False, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, attn_bias=None, attn_mask=None, cond_fn: Optional[Callable]=None):
        b = x.shape[0]
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)
        if exists(cond_fn):
            x = cond_fn(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        if exists(attn_bias):
            sim = sim + attn_bias
        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8, num_layers=2):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens
        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens), nn.GELU(), nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens))

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g=self.num_output_tokens)
        attn = self.net(x)
        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g=self.num_output_tokens)
        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / temperature ** omega
    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Dropsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConvResidual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lucidrains_robotic_transformer_pytorch(_paritybench_base):
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

