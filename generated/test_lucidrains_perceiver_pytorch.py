import sys
_module = sys.modules[__name__]
del sys
perceiver_pytorch = _module
experimental = _module
gated = _module
mixed_latents = _module
perceiver_io = _module
perceiver_pytorch = _module
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


import torch.nn.functional as F


from math import pi


from math import log


from functools import wraps


def exists(val):
    return val is not None


class LinearAttention(nn.Module):

    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q *= self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)
        if exists(mask):
            k.masked_fill_(mask, 0.0)
        context = einsum('b n d, b n e -> b d e', q, k)
        out = einsum('b d e, b n d -> b n e', context, v)
        out = rearrange(out, ' (b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


class Perceiver(nn.Module):

    def __init__(self, *, num_freq_bands, depth, max_freq, input_channels=3, input_axis=2, num_latents=512, latent_dim=512, cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64, num_classes=1000, attn_dropout=0.0, ff_dropout=0.0, weight_tie_layers=False, fourier_encode_data=True, self_per_cross_attn=1, final_classifier_head=True):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        fourier_channels = input_axis * (num_freq_bands * 2 + 1) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        get_cross_attn = lambda : PreNorm(latent_dim, Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda : PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda : PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda : PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self_attns = nn.ModuleList([])
            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([get_latent_attn(**cache_args, key=block_ind), get_latent_ff(**cache_args, key=block_ind)]))
            self.layers.append(nn.ModuleList([get_cross_attn(**cache_args), get_cross_ff(**cache_args), self_attns]))
        self.to_logits = nn.Sequential(Reduce('b n d -> b d', 'mean'), nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes)) if final_classifier_head else nn.Identity()

    def forward(self, data, mask=None, return_embeddings=False):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'
        if self.fourier_encode_data:
            axis_pos = list(map(lambda size: torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)
            data = torch.cat((data, enc_pos), dim=-1)
        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.latents, 'n d -> b n d', b=b)
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
        if return_embeddings:
            return x
        return self.to_logits(x)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class GRUGating(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, **kwargs):
        b, dim = x.shape[0], self.dim
        y = self.fn(x, **kwargs)
        gated_output = self.gru(rearrange(y, '... d -> (...) d'), rearrange(x, '... d -> (...) d'))
        gated_output = rearrange(gated_output, '(b n) d -> b n d', b=b)
        return gated_output


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)
    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices
    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    seq = seq[batch_indices, keep_indices]
    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, 'b -> b 1')
        mask = mask[batch_indices, keep_indices] & keep_mask
    return seq, mask


class PerceiverIO(nn.Module):

    def __init__(self, *, depth, dim, queries_dim, logits_dim=None, num_latents=512, latent_dim=512, cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64, weight_tie_layers=False, decoder_ff=False, seq_dropout_prob=0.0):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim), PreNorm(latent_dim, FeedForward(latent_dim))])
        get_latent_attn = lambda : PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda : PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for i in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)]))
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, queries=None):
        b, *_, device = *data.shape, data.device
        x = repeat(self.latents, 'n d -> b n d', b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        if self.training and self.seq_dropout_prob > 0.0:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)
        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        if not exists(queries):
            return x
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)
        latents = self.decoder_cross_attn(queries, context=x)
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)


class PerceiverLM(nn.Module):

    def __init__(self, *, dim, num_tokens, max_seq_len, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.perceiver_io = PerceiverIO(dim=dim, queries_dim=dim, logits_dim=num_tokens, **kwargs)

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb
        logits = self.perceiver_io(x, mask=mask, queries=x)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
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

class Test_lucidrains_perceiver_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

