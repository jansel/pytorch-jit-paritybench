import sys
_module = sys.modules[__name__]
del sys
criterions = _module
datasets = _module
data_utils = _module
fcdata = _module
kor_dataset = _module
kor_decompose = _module
nonpaired_dataset = _module
samplers = _module
thai_dataset = _module
thai_decompose = _module
evaluator = _module
inference = _module
logger = _module
models = _module
aux_classifier = _module
comp_encoder = _module
decoder = _module
discriminator = _module
ma_core = _module
memory = _module
modules = _module
blocks = _module
modules = _module
self_attention = _module
prepare_dataset = _module
ssim = _module
train = _module
trainer = _module
utils = _module
visualize = _module
writer = _module

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


import torch.nn.functional as F


import random


from itertools import product


import numpy as np


import torch


from torch.utils.data import Dataset


from itertools import chain


from torchvision import transforms


from torch.utils.data import DataLoader


from functools import partial


import torch.nn as nn


from math import exp


import torch.optim as optim


import copy


from torchvision import utils as tv_utils


class Flatten(nn.Module):

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


def dispatcher(dispatch_fn):

    def decorated(key, *args):
        if callable(key):
            return key
        if key is None:
            key = 'none'
        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def activ_dispatch(activ):
    return {'none': nn.Identity, 'relu': nn.ReLU, 'lrelu': partial(nn.LeakyReLU, negative_slope=0.2)}[activ.lower()]


@dispatcher
def norm_dispatch(norm):
    return {'none': nn.Identity, 'in': partial(nn.InstanceNorm2d, affine=False), 'bn': nn.BatchNorm2d}[norm.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {'zero': nn.ZeroPad2d, 'replicate': nn.ReplicationPad2d, 'reflect': nn.ReflectionPad2d}[pad_type.lower()]


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)


@dispatcher
def w_norm_dispatch(w_norm):
    return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


class ConvBlock(nn.Module):
    """Pre-activate conv block"""

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none', activ='relu', bias=True, upsample=False, downsample=False, w_norm='none', pad_type='zero', dropout=0.0):
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        activ = activ_dispatch(activ)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """Pre-activate residual block"""

    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False, norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.0):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ, upsample=upsample, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        out = out + x
        return out


class AuxClassifier(nn.Module):

    def __init__(self, C, C_out, norm='BN', activ='relu', pad_type='reflect', conv_dropout=0.0, clf_dropout=0.0):
        super().__init__()
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, dropout=conv_dropout)
        self.layers = nn.ModuleList([ResBlk(C, C * 2, 3, 1, downsample=True), ResBlk(C * 2, C * 2, 3, 1), nn.AdaptiveAvgPool2d(1), Flatten(1), nn.Dropout(clf_dropout), nn.Linear(C * 2, C_out)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GlobalContext(nn.Module):
    """ Global-context """

    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        C_bottleneck = int(C * bottleneck_ratio)
        w_norm = w_norm_dispatch(w_norm)
        self.k_proj = w_norm(nn.Conv2d(C, 1, 1))
        self.transform = nn.Sequential(w_norm(nn.Linear(C, C_bottleneck)), nn.LayerNorm(C_bottleneck), nn.ReLU(), w_norm(nn.Linear(C_bottleneck, C)))

    def forward(self, x):
        context_logits = self.k_proj(x)
        context_weights = F.softmax(context_logits.flatten(1), dim=1)
        context = torch.einsum('bci,bi->bc', x.flatten(2), context_weights)
        out = self.transform(context)
        return out[..., None, None]


class GCBlock(nn.Module):
    """ Global-context block """

    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        self.gc = GlobalContext(C, bottleneck_ratio, w_norm)

    def forward(self, x):
        gc = self.gc(x)
        return x + gc


class RelativePositionalEmbedding2d(nn.Module):
    """ Learned relative positional embedding
    return Q * (R_x + R_y) for input Q and learned embedding R
    """

    def __init__(self, emb_dim, H, W, down_kv=False):
        super().__init__()
        self.H = H
        self.W = W
        self.down_kv = down_kv
        self.h_emb = nn.Embedding(H * 2 - 1, emb_dim)
        self.w_emb = nn.Embedding(W * 2 - 1, emb_dim)
        rel_y, rel_x = self.rel_grid()
        self.register_buffer('rel_y', rel_y)
        self.register_buffer('rel_x', rel_x)

    def rel_grid(self):
        y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        rel_y = y.reshape(1, -1) - y.reshape(-1, 1)
        rel_x = x.reshape(1, -1) - x.reshape(-1, 1)
        if self.down_kv:

            def down(x):
                n_q, n_k = x.shape
                x = x.view(n_q, 1, int(n_k ** 0.5), int(n_k ** 0.5))
                return (F.avg_pool2d(x.float(), 2) - 0.5).flatten(1).long()
            rel_y = down(rel_y)
            rel_x = down(rel_x)
        rel_y += self.H - 1
        rel_x += self.W - 1
        return rel_y, rel_x

    def forward(self, query):
        """
        Args:
            query: [B, n_heads, C_qk, H*W]

        return:
            [B, n_heads, H*W, H*W]
        """
        r_x = self.w_emb(self.rel_x)
        r_y = self.h_emb(self.rel_y)
        S_rel = torch.einsum('bhci,ijc->bhij', query, r_x + r_y)
        return S_rel


def split_dim(x, dim, n_chunks):
    shape = x.shape
    assert shape[dim] % n_chunks == 0
    return x.view(*shape[:dim], n_chunks, shape[dim] // n_chunks, *shape[dim + 1:])


class Attention(nn.Module):

    def __init__(self, C_in_q, C_in_kv, C_qk, C_v, w_norm='none', scale=False, n_heads=1, down_kv=False, rel_pos_size=None):
        """
        Args:
            C_in_q: query source (encoder feature x)
            C_in_kv: key/value source (decoder feature y)
            C_qk: inner query/key dim, which should be same
            C_v: inner value dim, which same as output dim

            down_kv: Area attention for lightweight self-attention
                w/ mean pooling.
            rel_pos_size: height & width for relative positional embedding.
                If None or 0 is given, do not use relative positional embedding.
        """
        super().__init__()
        self.n_heads = n_heads
        self.down_kv = down_kv
        w_norm = w_norm_dispatch(w_norm)
        self.q_proj = w_norm(nn.Conv1d(C_in_q, C_qk, 1))
        self.k_proj = w_norm(nn.Conv1d(C_in_kv, C_qk, 1))
        self.v_proj = w_norm(nn.Conv1d(C_in_kv, C_v, 1))
        self.out = w_norm(nn.Conv2d(C_v, C_v, 1))
        if scale:
            self.scale = 1.0 / C_qk ** 0.5
        if rel_pos_size:
            C_h_qk = C_qk // n_heads
            self.rel_pos = RelativePositionalEmbedding2d(C_h_qk, rel_pos_size, rel_pos_size, down_kv=down_kv)

    def forward(self, x, y):
        """ Attend from x (decoder) to y (encoder)

        Args:
            x: decoder feature
            y: encoder feature
        """
        B, C, H, W = x.shape
        flat_x = x.flatten(start_dim=2)
        if not self.down_kv:
            flat_y = y.flatten(start_dim=2)
        else:
            y_down = F.avg_pool2d(y, 2)
            flat_y = y_down.flatten(2)
        query = self.q_proj(flat_x)
        key = self.k_proj(flat_y)
        value = self.v_proj(flat_y)
        query = split_dim(query, 1, self.n_heads)
        key = split_dim(key, 1, self.n_heads)
        value = split_dim(value, 1, self.n_heads)
        attn_score = torch.einsum('bhcq,bhck->bhqk', query, key)
        if hasattr(self, 'rel_pos'):
            attn_score += self.rel_pos(query)
        if hasattr(self, 'scale'):
            attn_score *= self.scale
        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = torch.einsum('bhqk,bhck->bhcq', attn_w, value).reshape(B, C, H, W)
        out = self.out(attn_out)
        return out


class AttentionFFNBlock(nn.Module):
    """ Transformer-like attention + ffn block """

    def __init__(self, C_in_q, C_in_kv, C_qk, C_v, size, scale=True, norm='ln', dropout=0.1, activ='relu', n_heads=1, ffn_mult=4, area=False, rel_pos=False):
        super().__init__()
        self.C_out = C_v
        if rel_pos:
            rel_pos = size
        self.attn = Attention(C_in_q, C_in_kv, C_qk, C_v, scale=scale, n_heads=n_heads, down_kv=area, rel_pos_size=rel_pos)
        self.dropout = nn.Dropout2d(dropout)
        self.ffn = nn.Sequential(ConvBlock(C_v, C_v * ffn_mult, 1, 1, 0, activ='none'), nn.Dropout2d(dropout), ConvBlock(C_v * ffn_mult, C_v, 1, 1, 0, activ=activ))
        if norm == 'ln':
            self.norm = nn.LayerNorm([C_v, size, size])
        else:
            norm = norm_dispatch(norm)
            self.norm = norm(C_v)

    def forward(self, x, y):
        skip = x
        x = self.norm(x)
        x = self.attn(x, y)
        x = self.dropout(x)
        x = self.ffn(x)
        x += skip
        return x


class SAFFNBlock(AttentionFFNBlock):

    def __init__(self, C, size, C_qk_ratio=0.25, scale=True, norm='ln', dropout=0.1, activ='relu', n_heads=1, ffn_mult=4, area=False, rel_pos=False):
        C_in_q = C
        C_in_kv = C
        C_qk = int(C * C_qk_ratio)
        C_v = C
        super().__init__(C_in_q, C_in_kv, C_qk, C_v, size, scale, norm, dropout, activ, n_heads, ffn_mult, area, rel_pos)
        self.C_in = C

    def forward(self, x):
        return super().forward(x, x)


class ComponentEncoder(nn.Module):
    """ Component image decomposer
    Encode the glyph into each component-wise features
    """

    def __init__(self, C_in, C, norm='none', activ='relu', pad_type='reflect', sa=None, n_comp_types=3):
        super().__init__()
        self.n_heads = n_comp_types
        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
        SAFFNBlk = partial(SAFFNBlock, **sa)
        self.shared = nn.ModuleList([ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'), ConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True), GCBlock(C * 2), ConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True), SAFFNBlk(C * 4, size=32, rel_pos=True)])
        self.heads = nn.ModuleList([nn.ModuleList([ResBlk(C * 4, C * 4, 3, 1), SAFFNBlk(C * 4, size=32, rel_pos=False), ResBlk(C * 4, C * 4, 3, 1), ResBlk(C * 4, C * 8, 3, 1, downsample=True), SAFFNBlk(C * 8, size=16, rel_pos=False), ResBlk(C * 8, C * 8)]) for _ in range(self.n_heads)])
        self.skip_layers = [3]
        self.final_shape = C * 8, 16, 16

    def forward(self, x):
        for layer in self.shared:
            x = layer(x)
        feats = [x]
        xs = [x] * self.n_heads
        n_layers = len(self.heads[0])
        for layer_idx in range(n_layers):
            for head_idx, head in enumerate(self.heads):
                layer = head[layer_idx]
                xs[head_idx] = layer(xs[head_idx])
            comp_feature = torch.stack(xs, dim=1)
            feats.append(comp_feature)
        return feats

    def filter_skips(self, feats):
        if self.skip_layers is None:
            return None
        return [feats[i] for i in self.skip_layers]


class Integrator(nn.Module):
    """Integrate component type-wise features"""

    def __init__(self, C, n_comps=3, norm='none', activ='none', C_in=None):
        super().__init__()
        C_in = (C_in or C) * n_comps
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps):
        """
        Args:
            comps [B, n_comps, C, H, W]: component features
        """
        inputs = comps.flatten(1, 2)
        out = self.integrate_layer(inputs)
        return out


class Upsample1x1(nn.Module):
    """Upsample 1x1 to 2x2 using Linear"""

    def __init__(self, C_in, C_out, norm='none', activ='relu', w_norm='none'):
        assert norm.lower() != 'in', 'Do not use instance norm for 1x1 spatial size'
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.proj = ConvBlock(C_in, C_out * 4, 1, 1, 0, norm=norm, activ=activ, w_norm=w_norm)

    def forward(self, x):
        x = self.proj(x)
        B, C = x.shape[:2]
        return x.view(B, C // 4, 2, 2)


class HourGlass(nn.Module):
    """U-net like hourglass module"""

    def __init__(self, C_in, C_max, size, n_downs, n_mids=1, norm='none', activ='relu', w_norm='none', pad_type='zero'):
        """
        Args:
            C_max: maximum C_out of left downsampling block's output
        """
        super().__init__()
        assert size == n_downs ** 2, 'HGBlock assume that the spatial size is downsampled to 1x1.'
        self.C_in = C_in
        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, w_norm=w_norm, pad_type=pad_type)
        self.lefts = nn.ModuleList()
        c_in = C_in
        for i in range(n_downs):
            c_out = min(c_in * 2, C_max)
            self.lefts.append(ConvBlk(c_in, c_out, downsample=True))
            c_in = c_out
        self.mids = nn.Sequential(*[ConvBlk(c_in, c_out, kernel_size=1, padding=0) for _ in range(n_mids)])
        self.rights = nn.ModuleList()
        for i, lb in enumerate(self.lefts[::-1]):
            c_out = lb.C_in
            c_in = lb.C_out
            channel_in = c_in * 2 if i else c_in
            if i == 0:
                block = Upsample1x1(channel_in, c_out, norm=norm, activ=activ, w_norm=w_norm)
            else:
                block = ConvBlk(channel_in, c_out, upsample=True)
            self.rights.append(block)

    def forward(self, x):
        features = []
        for lb in self.lefts:
            x = lb(x)
            features.append(x)
        assert x.shape[-2:] == torch.Size((1, 1))
        for i, (rb, lf) in enumerate(zip(self.rights, features[::-1])):
            if i:
                x = torch.cat([x, lf], dim=1)
            x = rb(x)
        return x


class Decoder(nn.Module):

    def __init__(self, C, C_out, size, norm='IN', activ='relu', pad_type='reflect', n_comp_types=3):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
        HGBlk = partial(HourGlass, size=size, norm='BN', activ=activ, pad_type=pad_type)
        IntegrateBlk = partial(Integrator, norm='none', activ='none', n_comps=n_comp_types)
        self.layers = nn.ModuleList([IntegrateBlk(C * 8), HGBlk(C * 8, C * 16, n_downs=4), ResBlk(C * 8, C * 8, 3, 1), ResBlk(C * 8, C * 8, 3, 1), ConvBlk(C * 8, C * 4, 3, 1, 1, upsample=True), ConvBlk(C * 12, C * 8, 3, 1, 1), ConvBlk(C * 8, C * 8, 3, 1, 1), ConvBlk(C * 8, C * 4, 3, 1, 1), ConvBlk(C * 4, C * 2, 3, 1, 1, upsample=True), ConvBlk(C * 2, C * 1, 3, 1, 1, upsample=True), ConvBlk(C * 1, C_out, 3, 1, 1)])
        self.skip_indices = [5]
        self.skip_layers = nn.ModuleList([IntegrateBlk(C * 8, C_in=C * 4)])
        self.out = nn.Tanh()

    def forward(self, comps, skips=None):
        """
        Args:
            comps [B, n_comps, C, H, W]: component features
            skips: skip features
        """
        if skips is not None:
            assert len(skips) == 1
            skip_idx = self.skip_indices[0]
            skip_layer = self.skip_layers[0]
            skip_feat = skips[0]
        x = comps
        for i, layer in enumerate(self.layers):
            if i == skip_idx:
                skip_feat = skip_layer(skip_feat)
                x = torch.cat([x, skip_feat], dim=1)
            x = layer(x)
        return self.out(x)


class MultitaskDiscriminator(nn.Module):

    def __init__(self, C, n_fonts, n_chars, use_rx=True, w_norm='spectral', activ='none'):
        super().__init__()
        self.use_rx = use_rx
        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))
        if use_rx:
            self.rx = w_norm(nn.Conv2d(C, 1, kernel_size=1, padding=0))

    def forward(self, x, font_indices, char_indices):
        """
        Args:
            x: [B, C, H, W]
            font_indices: [B]
            char_indices: [B]

        Return:
            [rx_logit, font_logit, char_logit]; [B, 1, H, W]
        """
        x = self.activ(x)
        font_emb = self.font_emb(font_indices)
        char_emb = self.char_emb(char_indices)
        if hasattr(self, 'rx'):
            rx_out = self.rx(x)
            ret = [rx_out]
        else:
            ret = [torch.as_tensor(0.0)]
        font_out = torch.einsum('bchw,bc->bhw', x, font_emb).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x, char_emb).unsqueeze(1)
        ret += [font_out, char_out]
        return ret


class Discriminator(nn.Module):

    def __init__(self, C, n_fonts, n_chars, activ='relu', gap_activ='relu', w_norm='spectral', use_rx=False, pad_type='reflect'):
        super().__init__()
        ConvBlk = partial(ConvBlock, w_norm=w_norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, w_norm=w_norm, activ=activ, pad_type=pad_type)
        feats = [ConvBlk(1, C, stride=2, activ='none'), ResBlk(C * 1, C * 2, downsample=True), ResBlk(C * 2, C * 4, downsample=True), ResBlk(C * 4, C * 8, downsample=True), ResBlk(C * 8, C * 16, downsample=False), ResBlk(C * 16, C * 32, downsample=False), ResBlk(C * 32, C * 32, downsample=False)]
        self.feats = nn.ModuleList(feats)
        gap_activ = activ_dispatch(gap_activ)
        self.gap = nn.Sequential(gap_activ(), nn.AdaptiveAvgPool2d(1))
        self.projD = MultitaskDiscriminator(C * 32, n_fonts, n_chars, use_rx=use_rx, w_norm=w_norm)

    def forward(self, x, font_indices, char_indices, out_feats=False):
        feats = []
        for layer in self.feats:
            x = layer(x)
            feats.append(x)
        x = self.gap(x)
        ret = self.projD(x, font_indices, char_indices)
        if out_feats:
            ret.append(feats)
        return ret

    @property
    def use_rx(self):
        return self.projD.use_rx


class DynamicMemory(nn.Module):

    def __init__(self):
        super().__init__()
        self.memory = {}
        self.reset()

    def write(self, style_ids, comp_addrs, comp_feats):
        """ Batch write

        Args:
            style_ids: [B]
            comp_addrs: [B, 3]
            comp_feats: [B, 3, mem_shape]
        """
        assert len(style_ids) == len(comp_addrs) == len(comp_feats), 'Input sizes are different'
        for style_id, comp_addrs_per_char, comp_feats_per_char in zip(style_ids, comp_addrs, comp_feats):
            for comp_addr, comp_feat in zip(comp_addrs_per_char, comp_feats_per_char):
                self.write_point(style_id, comp_addr, comp_feat)

    def read(self, style_ids, comp_addrs, reduction='mean'):
        """ Batch read

        Args:
            style_ids: [B]
            comp_addrs: [B, 3]
            reduction: reduction method if multiple features exist in sample memory address:
                       ['mean' (default), 'first', 'rand', 'none']
        """
        out = []
        for style_id, comp_addrs_per_char in zip(style_ids, comp_addrs):
            char_feats = []
            for comp_addr in comp_addrs_per_char:
                comp_feat = self.read_point(style_id, comp_addr, reduction)
                char_feats.append(comp_feat)
            char_feats = torch.stack(char_feats)
            out.append(char_feats)
        out = torch.stack(out)
        return out

    def write_point(self, style_id, comp_addr, data):
        self.memory.setdefault(style_id.item(), {}).setdefault(comp_addr.item(), []).append(data)

    def read_point(self, style_id, comp_addr, reduction='mean'):
        """ Point read """
        comp_feats = self.memory[style_id.item()][comp_addr.item()]
        return self.reduce_features(comp_feats, reduction)

    def reduce_features(self, feats, reduction='mean'):
        if len(feats) == 1:
            return feats[0]
        if reduction == 'mean':
            return torch.stack(feats).mean(dim=0)
        elif reduction == 'first':
            return feats[0]
        elif reduction == 'rand':
            return np.random.choice(feats)
        elif reduction == 'none':
            return feats
        else:
            raise ValueError(reduction)

    def reset(self):
        self.memory = {}

    def reset_batch(self, style_ids, comp_addrs):
        for style_id, comp_addrs_per_char in zip(style_ids, comp_addrs):
            for comp_addr in comp_addrs_per_char:
                self.reset_point(style_id, comp_addr)

    def reset_point(self, style_id, comp_addr):
        self.memory[style_id.item()].pop(comp_addr.item())


class PersistentMemory(nn.Module):

    def __init__(self, n_comps, mem_shape):
        """
        Args:
            mem_shape: (C, H, W) tuple (3-elem)
        """
        super().__init__()
        self.shape = mem_shape
        self.bias = nn.Parameter(torch.randn(n_comps, *mem_shape))
        C = mem_shape[0]
        self.hypernet = nn.Sequential(ConvBlock(C, C), ConvBlock(C, C), ConvBlock(C, C))

    def read(self, comp_addrs):
        b = self.bias[comp_addrs]
        return b

    def forward(self, x, comp_addrs):
        """
        Args:
            x: [B, 3, *mem_shape]
            comp_addr: [B, 3]
        """
        b = self.read(comp_addrs)
        B = b.size(0)
        b = b.flatten(0, 1)
        b = self.hypernet(b)
        b = split_dim(b, 0, B)
        return x + b


def comp_id_to_addr(ids, language):
    """ Component id to memory address converter

    Args:
        ids [B, 3 or 4], torch.tensor: [B, 3] -> kor, [B, 4] -> thai.
    """
    ids = ids.clone()
    if language == 'kor':
        ids[:, 1] += kor.N_CHO
        ids[:, 2] += kor.N_CHO + kor.N_JUNG
    elif language == 'thai':
        ids[:, 1] += thai.N_CONSONANTS
        ids[:, 2] += thai.N_CONSONANTS + thai.N_UPPERS
        ids[:, 3] += thai.N_CONSONANTS + thai.N_UPPERS + thai.N_HIGHESTS
    else:
        raise ValueError(language)
    return ids


class Memory(nn.Module):
    STYLE_ADDR = -1

    def __init__(self, mem_shape, n_comps, persistent, language):
        """
        Args:
            mem_shape (tuple [3]):
                memory shape in (C, H, W) tuple, which is same as encoded feature shape
            n_comps: # of total components, which identify persistent memory size
        """
        super().__init__()
        self.dynamic_memory = DynamicMemory()
        self.mem_shape = mem_shape
        self.persistent = persistent
        self.language = language
        if persistent:
            self.persistent_memory = PersistentMemory(n_comps, mem_shape)

    def write(self, style_ids, comp_ids, comp_feats):
        """ Write data into dynamic memory """
        comp_addrs = comp_id_to_addr(comp_ids, self.language)
        self.dynamic_memory.write(style_ids, comp_addrs, comp_feats)

    def read(self, style_ids, comp_ids):
        """ Read data from memory (dynamic w/ or w/o persistent)

        Args:
            comp_ids [B, 3]
        """
        comp_addrs = comp_id_to_addr(comp_ids, self.language)
        mem = self.dynamic_memory.read(style_ids, comp_addrs)
        if self.persistent:
            mem = self.persistent_memory(mem, comp_addrs)
        return mem

    def reset_style(self, style_ids):
        style_addrs = self.get_style_addr(len(style_ids))
        self.dynamic_memory.reset_batch(style_ids, style_addrs)

    def write_style(self, style_ids, style_codes):
        style_addrs = self.get_style_addr(len(style_ids))
        self.dynamic_memory.write(style_ids, style_addrs, style_codes.unsqueeze(1))

    def read_style(self, style_ids):
        style_addrs = self.get_style_addr(len(style_ids))
        return self.dynamic_memory.read(style_ids, style_addrs).squeeze(1)

    def get_style_addr(self, N):
        return torch.full([N, 1], self.STYLE_ADDR, dtype=torch.long)

    def reset_dynamic(self):
        """ Reset dynamic memory """
        self.dynamic_memory.reset()


class MACore(nn.Module):
    """ Memory-augmented HFG """

    def __init__(self, C_in, C, C_out, comp_enc, dec, n_comps, n_comp_types, language):
        """
        Args:
            C_in: 1
            C: unit of channel size
            C_out: 1

            comp_enc: component encoder configs
            dec: decoder configs

            n_comps: # of total component instances.
            n_comp_types: # of component types. kor=3, thai=4.
        """
        super().__init__()
        self.component_encoder = ComponentEncoder(C_in, C, **comp_enc, n_comp_types=n_comp_types)
        self.mem_shape = self.component_encoder.final_shape
        self.memory = Memory(self.mem_shape, n_comps, persistent=True, language=language)
        if self.component_encoder.skip_layers is not None:
            self.skip_memory = Memory(self.mem_shape, n_comps, persistent=False, language=language)
            skip_layers = self.component_encoder.skip_layers
            assert skip_layers is None or len(skip_layers) == 1, 'Only supports #skip_layers <= 1'
        self.decoder = Decoder(C, C_out, self.mem_shape[-1], **dec, n_comp_types=n_comp_types)

    def reset_dynamic_memory(self):
        self.memory.reset_dynamic()
        if hasattr(self, 'skip_memory'):
            self.skip_memory.reset_dynamic()

    def encode_write(self, style_ids, comp_ids, style_imgs, reset_dynamic_memory=True):
        """ Encode feature from input data and write it to memory
        Args:
            # batch size B can be different with infer phase
            style_ids [B]: style index
            comp_ids [B, n_comp_types]: component ids of style chars
            style_imgs [B, 1, 128, 128]: eq_fonts
        """
        if reset_dynamic_memory:
            self.reset_dynamic_memory()
        feats = self.component_encoder(style_imgs)
        comp_feats = feats[-1]
        skips = self.component_encoder.filter_skips(feats)
        self.memory.write(style_ids, comp_ids, comp_feats)
        if hasattr(self, 'skip_memory'):
            self.skip_memory.write(style_ids, comp_ids, skips[0])
        return comp_feats

    def read_decode(self, target_style_ids, target_comp_ids):
        """ Read feature from memory and decode it
        Args:
            # batch size B can be different with write phase
            target_style_ids: [B]
            target_comp_ids: [B, n_comp_types]
        """
        comp_feats = self.memory.read(target_style_ids, target_comp_ids)
        skips = None
        if hasattr(self, 'skip_memory'):
            skip_feats = self.skip_memory.read(target_style_ids, target_comp_ids)
            skips = [skip_feats]
        out = self.decoder(comp_feats, skips)
        return out


class LinearBlock(nn.Module):
    """Pre-activate linear block"""

    def __init__(self, C_in, C_out, norm='none', activ='relu', bias=True, w_norm='none', dropout=0.0):
        super().__init__()
        activ = activ_dispatch(activ)
        if norm.lower() == 'bn':
            norm = nn.BatchNorm1d
        elif norm.lower() == 'none':
            norm = nn.Identity
        else:
            raise ValueError(f'LinearBlock supports BN only (but {norm} is given)')
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, val_range=None, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        window = create_window(window_size, channel)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        assert self.channel == img1.size(1)
        return ssim(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range)


def msssim(img1, img2, weights=None, window_size=11, window=None, size_average=True, val_range=None, normalize=False):
    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, window=window, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class MSSSIM(torch.nn.Module):

    def __init__(self, weights=None, window_size=11, size_average=True, val_range=None, channel=1, normalize=False):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        self.normalize = normalize
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.register_buffer('weights', torch.as_tensor(weights))
        window = create_window(window_size, channel)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        assert img1.size(1) == self.channel
        return msssim(img1, img2, weights=self.weights, window_size=self.window_size, window=self.window, size_average=self.size_average, val_range=self.val_range, normalize=self.normalize)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'C_in_q': 4, 'C_in_kv': 4, 'C_qk': 4, 'C_v': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionFFNBlock,
     lambda: ([], {'C_in_q': 4, 'C_in_kv': 4, 'C_qk': 4, 'C_v': 4, 'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AuxClassifier,
     lambda: ([], {'C': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCBlock,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalContext,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SAFFNBlock,
     lambda: ([], {'C': 4, 'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_clovaai_dmfont(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

