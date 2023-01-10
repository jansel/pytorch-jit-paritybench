import sys
_module = sys.modules[__name__]
del sys
coca_pytorch = _module
coca_pytorch = _module
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


from torch import einsum


from torch import nn


import torch.nn.functional as F


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

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class EmbedToLatents(nn.Module):

    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


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


def exists(val):
    return val is not None


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

    def forward(self, x, attn_mask=None):
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
        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.attn_out(out) + self.ff_out(ff)


def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):

    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, parallel_ff=False, ff_mult=4, norm_context=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        ff_inner_dim = ff_mult * dim
        self.ff = nn.Sequential(nn.Linear(dim, ff_inner_dim * 2, bias=False), SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False)) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        x = self.norm(x)
        context = self.context_norm(context)
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale
        k, v = self.to_kv(context).chunk(2, dim=-1)
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if exists(self.ff):
            out = out + self.ff(x)
        return out


class CoCa(nn.Module):

    def __init__(self, *, dim, num_tokens, unimodal_depth, multimodal_depth, dim_latents=None, image_dim=None, num_img_queries=256, dim_head=64, heads=8, ff_mult=4, img_encoder=None, caption_loss_weight=1.0, contrastive_loss_weight=1.0, pad_id=0):
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))
        self.img_encoder = img_encoder
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim))
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(dim)
        self.text_cls_norm = LayerNorm(dim)
        dim_latents = default(dim_latents, dim)
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        self.text_to_latents = EmbedToLatents(dim, dim_latents)
        self.temperature = nn.Parameter(torch.Tensor([1.0]))
        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)))
        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)), Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))]))
        self.to_logits = nn.Sequential(LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False))
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def embed_text(self, text):
        batch, device = text.shape[0], text.device
        seq = text.shape[1]
        text_tokens = self.token_emb(text)
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)
        cls_mask = rearrange(text != self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)
        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens

    def embed_image(self, images=None, image_tokens=None):
        assert not (exists(images) and exists(image_tokens))
        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images)
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)
        return img_queries[:, 0], img_queries[:, 1:]

    def forward(self, text, images=None, image_tokens=None, labels=None, return_loss=False, return_embeddings=False):
        batch, device = text.shape[0], text.device
        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]
        text_embeds, text_tokens = self.embed_text(text)
        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)
        if return_embeddings:
            return text_embeds, image_embeds
        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)
        logits = self.to_logits(text_tokens)
        if not return_loss:
            return logits
        ce = F.cross_entropy
        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight
        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)
        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)
        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight
        return caption_loss + contrastive_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EmbedToLatents,
     lambda: ([], {'dim': 4, 'dim_latents': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
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
]

class Test_lucidrains_CoCa_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

