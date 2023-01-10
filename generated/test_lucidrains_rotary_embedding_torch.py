import sys
_module = sys.modules[__name__]
del sys
rotary_embedding_torch = _module
rotary_embedding_torch = _module
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


from math import pi


from math import log


import torch


from torch import nn


from torch import einsum


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = t * freqs.cos() * scale + rotate_half(t) * freqs.sin() * scale
    return torch.cat((t_left, t, t_right), dim=-1)


def exists(val):
    return val is not None


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, learned_freq=False, use_xpos=False, xpos_scale_base=512):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.register_buffer('scale', scale)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        device, seq_len = t.device, t.shape[seq_dim]
        freqs = self.forward(lambda : torch.arange(seq_len, device=device), cache_key=seq_len)
        return apply_rotary_emb(freqs, t)

    def rotate_queries_and_keys(self, q, k, seq_dim=-2):
        assert self.use_xpos
        device, seq_len = q.device, q.shape[seq_dim]
        seq = torch.arange(seq_len, device=device)
        freqs = self.forward(lambda : seq, cache_key=seq_len)
        scale = self.get_scale(lambda : seq, cache_key=seq_len)
        rotated_q = apply_rotary_emb(freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1)
        return rotated_q, rotated_k

    def get_scale(self, t, cache_key=None):
        assert self.use_xpos
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]
        if callable(t):
            t = t()
        scale = 1.0
        if self.use_xpos:
            power = t - len(t) // 2 / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)
        if exists(cache_key):
            self.cache[cache_key] = freqs
        return scale

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]
        if callable(t):
            t = t()
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        if exists(cache_key):
            self.cache[cache_key] = freqs
        return freqs

