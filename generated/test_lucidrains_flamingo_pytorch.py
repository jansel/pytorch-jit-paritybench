import sys
_module = sys.modules[__name__]
del sys
flamingo_pytorch = _module
flamingo_palm = _module
flamingo_pytorch = _module
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


from torch import einsum


from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class ParallelTransformerBlock(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = attn_inner_dim, dim_head, dim_head, ff_inner_dim * 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))
        self.register_buffer('mask', None, persistent=False)
        self.register_buffer('pos_emb', None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]
        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer('pos_emb', pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.attn_out(out) + self.ff_out(ff)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False))


def exists(val):
    return val is not None


class MaskedCrossAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, only_attend_immediate_media=True):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        b, t, m = media.shape[:3]
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        if exists(media_locations):
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(t, device=x.device) + 1
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m=m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        if exists(media_locations) and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.0)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4, only_attend_immediate_media=True):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, media, media_locations=None):
        x = self.attn(x, media, media_locations=media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):

    def __init__(self, *, dim, depth, dim_head=64, heads=8, num_latents=64, num_media_embeds=4, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        times = x.shape[1]
        x = x + self.media_pos_emb[:times]
        latents = repeat(self.latents, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


class FlamingoPaLM(nn.Module):

    def __init__(self, *, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4, media_token_id=3, cross_attn_every=3, img_encoder=None, perceiver_num_latents=64, perceiver_depth=2, max_video_frames=None, only_attend_immediate_media=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.media_token_id = media_token_id
        self.video_frame_pos_emb = nn.Parameter(torch.randn(max_video_frames, dim)) if exists(max_video_frames) else None
        self.img_encoder = img_encoder
        freeze_model_and_make_eval_(self.img_encoder)
        self.perceiver_resampler = PerceiverResampler(dim=dim, depth=perceiver_depth, dim_head=dim_head, heads=heads, num_latents=perceiver_num_latents)
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)), GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media) if not ind % cross_attn_every else None]))
        self.to_logits = nn.Sequential(LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False))
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, text, *, images=None, videos=None, embeds=None):
        batch, device = text.shape[0], text.device
        flamingo_mode = any([exists(t) for t in (images, videos, embeds)])
        if flamingo_mode:
            freeze_all_layers_(self)
            unfreeze_all_layers_(self.perceiver_resampler)
            [unfreeze_all_layers_(cross_attn) for _, cross_attn in self.layers if exists(cross_attn)]
        else:
            unfreeze_all_layers_(self)
        if flamingo_mode:
            media_locations = text == self.media_token_id
        text_tokens = self.token_emb(text)
        assert not (exists(embeds) and (exists(images) or exists(video)))
        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            images = rearrange(images, 'b t ... -> (b t) ...')
            with torch.no_grad():
                embeds = self.img_encoder(images)
            embeds = rearrange(embeds, '(b t) ... -> b t ...', b=batch)
        if exists(videos):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic video encoding'
            batch, media, num_times, *_ = videos.shape
            videos = rearrange(videos, '... c h w -> (...) c h w')
            with torch.no_grad():
                embeds = self.img_encoder(videos)
            embeds = rearrange(embeds, '(b m t) ... -> b m t ...', b=batch, m=media, t=num_times)
            video_time_pos_emb = repeat(self.video_frame_pos_emb[:num_times], 't d -> b m t n d', b=batch, m=media, n=embeds.shape[-2])
            embeds = embeds + video_time_pos_emb
            embeds = rearrange(embeds, 'b m t n d -> b m (t n) d')
        if exists(embeds):
            embeds = self.perceiver_resampler(embeds)
        for attn_ff, flamingo_cross_attn in self.layers:
            text_tokens = attn_ff(text_tokens)
            if exists(flamingo_cross_attn) and exists(embeds):
                text_tokens = flamingo_cross_attn(text_tokens, embeds, media_locations=media_locations)
        return self.to_logits(text_tokens)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwiGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lucidrains_flamingo_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

