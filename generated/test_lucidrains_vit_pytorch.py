import sys
_module = sys.modules[__name__]
del sys
setup = _module
test = _module
vit_pytorch = _module
ats_vit = _module
cait = _module
cct = _module
cct_3d = _module
cross_vit = _module
crossformer = _module
cvt = _module
deepvit = _module
dino = _module
distill = _module
efficient = _module
es_vit = _module
extractor = _module
learnable_memory_vit = _module
levit = _module
local_vit = _module
mae = _module
max_vit = _module
mobile_vit = _module
mpp = _module
nest = _module
parallel_vit = _module
pit = _module
recorder = _module
regionvit = _module
rvt = _module
scalable_vit = _module
sep_vit = _module
simmim = _module
simple_vit = _module
simple_vit_1d = _module
simple_vit_3d = _module
simple_vit_with_patch_dropout = _module
t2t = _module
twins_svt = _module
vit = _module
vit_1d = _module
vit_3d = _module
vit_for_small_dataset = _module
vit_with_patch_dropout = _module
vit_with_patch_merger = _module
vivit = _module

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


from torch.nn.utils.rnn import pad_sequence


from torch import nn


from torch import einsum


from random import randrange


import copy


import random


from functools import wraps


from functools import partial


from torchvision import transforms as T


from math import ceil


from math import sqrt


import torch.nn as nn


import math


from math import pi


from math import log


def sample_gumbel(shape, device, dtype, eps=1e-06):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


class AdaptiveTokenSampling(nn.Module):

    def __init__(self, output_num_tokens, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.device, attn.dtype
        cls_attn = attn[..., 0, 1:]
        value_norms = value[..., 1:, :].norm(dim=-1)
        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)
        normed_cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + eps)
        pseudo_logits = log(normed_cls_attn)
        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k=output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device=device, dtype=dtype)
        sampled_token_ids = pseudo_logits.argmax(dim=-1) + 1
        unique_sampled_token_ids_list = [torch.unique(t, sorted=True) for t in torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first=True)
        new_mask = unique_sampled_token_ids != 0
        new_mask = F.pad(new_mask, (1, 0), value=True)
        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value=0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h=heads)
        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim=2)
        return new_attn, new_mask, unique_sampled_token_ids


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


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


def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):

    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, spatial_depth, temporal_depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        num_image_patches = image_height // patch_height * (image_width // patch_width)
        num_frame_patches = frames // frame_patch_size
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.global_average_pool = pool == 'mean'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape
        x = x + self.pos_embedding
        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b=b, f=f)
            x = torch.cat((spatial_cls_tokens, x), dim=2)
        x = self.dropout(x)
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) n d -> b f n d', b=b)
        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b)
            x = torch.cat((temporal_cls_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')
        x = self.to_latent(x)
        return self.mlp_head(x)


class LayerScale(nn.Module):

    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-05
        else:
            init_eps = 1e-06
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class CaiT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, cls_depth, heads, mlp_dim, dim_head=64, dropout=0.0, emb_dropout=0.0, layer_dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.patch_transformer(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = self.cls_transformer(cls_tokens, context=x)
        return self.mlp_head(x[:, 0])


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype
        if drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = batch, *((1,) * (x.ndim - 1))
        keep_mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout, projection_dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class Tokenizer(nn.Module):

    def __init__(self, frame_kernel_size, kernel_size, stride, padding, frame_stride=1, frame_pooling_stride=1, frame_pooling_kernel_size=1, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, n_conv_layers=1, n_input_channels=3, n_output_channels=64, in_planes=64, activation=None, max_pool=True, conv_bias=False):
        super().__init__()
        n_filter_list = [n_input_channels] + [in_planes for _ in range(n_conv_layers - 1)] + [n_output_channels]
        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])
        self.conv_layers = nn.Sequential(*[nn.Sequential(nn.Conv3d(chan_in, chan_out, kernel_size=(frame_kernel_size, kernel_size, kernel_size), stride=(frame_stride, stride, stride), padding=(frame_kernel_size // 2, padding, padding), bias=conv_bias), nn.Identity() if not exists(activation) else activation(), nn.MaxPool3d(kernel_size=(frame_pooling_kernel_size, pooling_kernel_size, pooling_kernel_size), stride=(frame_pooling_stride, pooling_stride, pooling_stride), padding=(frame_pooling_kernel_size // 2, pooling_padding, pooling_padding)) if max_pool else nn.Identity()) for chan_in, chan_out in n_filter_list_pairs])
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, frames=8, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, frames, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        return rearrange(x, 'b c f h w -> b (f h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[(p / 10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class TransformerClassifier(nn.Module):

    def __init__(self, seq_pool=True, embedding_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout_rate=0.1, attention_dropout=0.1, stochastic_depth_rate=0.1, positional_embedding='sine', sequence_length=None, *args, **kwargs):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        assert exists(sequence_length) or positional_embedding == 'none', f'Positional embedding is set to {positional_embedding} and the sequence length was not specified.'
        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))
        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_rate, attention_dropout=attention_dropout, drop_path_rate=layer_dpr) for layer_dpr in dpr])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_token, x), dim=1)
        if exists(self.positional_emb):
            x += self.positional_emb
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim=1), x)
        else:
            x = x[:, 0]
        return self.fc(x)


class CCT(nn.Module):

    def __init__(self, img_size=224, num_frames=8, embedding_dim=768, n_input_channels=3, n_conv_layers=1, frame_stride=1, frame_kernel_size=3, frame_pooling_kernel_size=1, frame_pooling_stride=1, kernel_size=7, stride=2, padding=3, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, *args, **kwargs):
        super().__init__()
        img_height, img_width = pair(img_size)
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels, n_output_channels=embedding_dim, frame_stride=frame_stride, frame_kernel_size=frame_kernel_size, frame_pooling_stride=frame_pooling_stride, frame_pooling_kernel_size=frame_pooling_kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, pooling_kernel_size=pooling_kernel_size, pooling_stride=pooling_stride, pooling_padding=pooling_padding, max_pool=True, activation=nn.ReLU, n_conv_layers=n_conv_layers, conv_bias=False)
        self.classifier = TransformerClassifier(*args, sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels, frames=num_frames, height=img_height, width=img_width), embedding_dim=embedding_dim, seq_pool=True, dropout_rate=0.0, attention_dropout=0.1, stochastic_depth=0.1, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class ProjectInOut(nn.Module):

    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class CrossTransformer(nn.Module):

    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))), ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)))]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


class MultiScaleEncoder(nn.Module):

    def __init__(self, *, depth, sm_dim, lg_dim, sm_enc_params, lg_enc_params, cross_attn_heads, cross_attn_depth, cross_attn_dim_head=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params), Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params), CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth, heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout)]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens


class ImageEmbedder(nn.Module):

    def __init__(self, *, dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        return self.dropout(x)


class CrossViT(nn.Module):

    def __init__(self, *, image_size, num_classes, sm_dim, lg_dim, sm_patch_size=12, sm_enc_depth=1, sm_enc_heads=8, sm_enc_mlp_dim=2048, sm_enc_dim_head=64, lg_patch_size=16, lg_enc_depth=4, lg_enc_heads=8, lg_enc_mlp_dim=2048, lg_enc_dim_head=64, cross_attn_depth=2, cross_attn_heads=8, cross_attn_dim_head=64, depth=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size, dropout=emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size, dropout=emb_dropout)
        self.multi_scale_encoder = MultiScaleEncoder(depth=depth, sm_dim=sm_dim, lg_dim=lg_dim, cross_attn_heads=cross_attn_heads, cross_attn_dim_head=cross_attn_dim_head, cross_attn_depth=cross_attn_depth, sm_enc_params=dict(depth=sm_enc_depth, heads=sm_enc_heads, mlp_dim=sm_enc_mlp_dim, dim_head=sm_enc_dim_head), lg_enc_params=dict(depth=lg_enc_depth, heads=lg_enc_heads, mlp_dim=lg_enc_mlp_dim, dim_head=lg_enc_dim_head), dropout=dropout)
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))
        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)
        return sm_logits + lg_logits


class CrossEmbedLayer(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / 2 ** i) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else (val,) * length


class CrossFormer(nn.Module):

    def __init__(self, *, dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), global_window_size=(8, 4, 2, 1), local_window_size=7, cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)), cross_embed_strides=(4, 2, 2, 2), num_classes=1000, attn_dropout=0.0, ff_dropout=0.0, channels=3):
        super().__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)
        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4
        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList([])
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
            self.layers.append(nn.ModuleList([CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride), Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers, attn_dropout=attn_dropout, ff_dropout=ff_dropout)]))
        self.to_logits = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(last_dim, num_classes))

    def forward(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        return self.to_logits(x)


class DepthWiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias), nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias))

    def forward(self, x):
        return self.net(x)


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class CvT(nn.Module):

    def __init__(self, *, num_classes, s1_emb_dim=64, s1_emb_kernel=7, s1_emb_stride=4, s1_proj_kernel=3, s1_kv_proj_stride=2, s1_heads=1, s1_depth=1, s1_mlp_mult=4, s2_emb_dim=192, s2_emb_kernel=3, s2_emb_stride=2, s2_proj_kernel=3, s2_kv_proj_stride=2, s2_heads=3, s2_depth=2, s2_mlp_mult=4, s3_emb_dim=384, s3_emb_kernel=3, s3_emb_stride=2, s3_proj_kernel=3, s3_kv_proj_stride=2, s3_heads=6, s3_depth=10, s3_mlp_mult=4, dropout=0.0):
        super().__init__()
        kwargs = dict(locals())
        dim = 3
        layers = []
        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            layers.append(nn.Sequential(nn.Conv2d(dim, config['emb_dim'], kernel_size=config['emb_kernel'], padding=config['emb_kernel'] // 2, stride=config['emb_stride']), LayerNorm(config['emb_dim']), Transformer(dim=config['emb_dim'], proj_kernel=config['proj_kernel'], kv_proj_stride=config['kv_proj_stride'], depth=config['depth'], heads=config['heads'], mlp_mult=config['mlp_mult'], dropout=dropout)))
            dim = config['emb_dim']
        self.layers = nn.Sequential(*layers)
        self.to_logits = nn.Sequential(nn.AdaptiveAvgPool2d(1), Rearrange('... () () -> ...'), nn.Linear(dim, num_classes))

    def forward(self, x):
        latents = self.layers(x)
        return self.to_logits(latents)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class DeepViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class RandomApply(nn.Module):

    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class L2Norm(nn.Module):

    def forward(self, x, eps=1e-06):
        return F.normalize(x, dim=1, eps=eps)


class MLP(nn.Module):

    def __init__(self, dim, dim_out, num_layers, hidden_size=256):
        super().__init__()
        layers = []
        dims = dim, *((hidden_size,) * (num_layers - 1))
        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == len(dims) - 1
            layers.extend([nn.Linear(layer_dim_in, layer_dim_out), nn.GELU() if not is_last else nn.Identity()])
        self.net = nn.Sequential(*layers, L2Norm(), nn.Linear(hidden_size, dim_out))

    def forward(self, x):
        return self.net(x)


def singleton(cache_key):

    def inner_fn(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class NetWrapper(nn.Module):

    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.view_projector = None
        self.region_projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('view_projector')
    def _get_view_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector

    @singleton('region_projector')
    def _get_region_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector

    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()
        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True):
        region_latents = self.get_embedding(x)
        global_latent = reduce(region_latents, 'b c h w -> b c', 'mean')
        if not return_projection:
            return global_latent, region_latents
        view_projector = self._get_view_projector(global_latent)
        region_projector = self._get_region_projector(region_latents)
        region_latents = rearrange(region_latents, 'b c h w -> b (h w) c')
        return view_projector(global_latent), region_projector(region_latents), region_latents


class EMA:

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def default(val, d):
    return val if exists(val) else d


def get_module_device(module):
    return next(module.parameters()).device


def loss_fn(teacher_logits, student_logits, teacher_temp, student_temp, centers, eps=1e-20):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim=-1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim=-1)
    return -(teacher_probs * torch.log(student_probs + eps)).sum(dim=-1).mean()


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Dino(nn.Module):

    def __init__(self, net, image_size, hidden_layer=-2, projection_hidden_size=256, num_classes_K=65336, projection_layers=4, student_temp=0.9, teacher_temp=0.04, local_upper_crop_scale=0.4, global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.9, augment_fn=None, augment_fn2=None):
        super().__init__()
        self.net = net
        DEFAULT_AUG = torch.nn.Sequential(RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3), T.RandomGrayscale(p=0.2), T.RandomHorizontalFlip(), RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2), T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)
        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale=(0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale=(global_lower_crop_scale, 1.0))
        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer=hidden_layer)
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)
        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers', torch.zeros(1, num_classes_K))
        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        device = get_module_device(net)
        self
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(self, x, return_embedding=False, return_projection=True, student_temp=None, teacher_temp=None):
        if return_embedding:
            return self.student_encoder(x, return_projection=return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)
        local_image_one, local_image_two = self.local_crop(image_one), self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)
        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)
        loss_fn_ = partial(loss_fn, student_temp=default(student_temp, self.student_temp), teacher_temp=default(teacher_temp, self.teacher_temp), centers=self.teacher_centers)
        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim=0)
        self.last_teacher_centers.copy_(teacher_logits_avg)
        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss


class DistillMixin:

    def forward(self, img, distill_token=None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b=b)
            x = torch.cat((x, distill_tokens), dim=1)
        x = self._attend(x)
        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        out = self.mlp_head(x)
        if distilling:
            return out, distill_tokens
        return out


class DistillableViT(DistillMixin, ViT):

    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class RearrangeImage(nn.Module):

    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


def conv_output_size(image_size, kernel_size, stride, padding):
    return int((image_size - kernel_size + 2 * padding) / stride + 1)


class T2TViT(nn.Module):

    def __init__(self, *, image_size, num_classes, dim, depth=None, heads=None, mlp_dim=None, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, transformer=None, t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        layers = []
        layer_dim = channels
        output_image_size = image_size
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == len(t2t_layers) - 1
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            layers.extend([RearrangeImage() if not is_first else nn.Identity(), nn.Unfold(kernel_size=kernel_size, stride=stride, padding=stride // 2), Rearrange('b c n -> b n c'), Transformer(dim=layer_dim, heads=1, depth=1, dim_head=layer_dim, mlp_dim=layer_dim, dropout=dropout) if not is_last else nn.Identity()])
        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class DistillableT2TViT(DistillMixin, T2TViT):

    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class DistillWrapper(nn.Module):

    def __init__(self, *, teacher, student, temperature=1.0, alpha=0.5, hard=False):
        super().__init__()
        assert isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT)), 'student must be a vision transformer'
        self.teacher = teacher
        self.student = student
        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard
        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))
        self.distill_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img, labels, temperature=None, alpha=None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        student_logits, distill_tokens = self.student(img, distill_token=self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)
        loss = F.cross_entropy(student_logits, labels)
        if not self.hard:
            distill_loss = F.kl_div(F.log_softmax(distill_logits / T, dim=-1), F.softmax(teacher_logits / T, dim=-1).detach(), reduction='batchmean')
            distill_loss *= T ** 2
        else:
            teacher_labels = teacher_logits.argmax(dim=-1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)
        return loss * (1 - alpha) + distill_loss * alpha


def region_loss_fn(teacher_logits, student_logits, teacher_latent, student_latent, teacher_temp, student_temp, centers, eps=1e-20):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim=-1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim=-1)
    sim_matrix = einsum('b i d, b j d -> b i j', student_latent, teacher_latent)
    sim_indices = sim_matrix.max(dim=-1).indices
    sim_indices = repeat(sim_indices, 'b n -> b n k', k=teacher_probs.shape[-1])
    max_sim_teacher_probs = teacher_probs.gather(1, sim_indices)
    return -(max_sim_teacher_probs * log(student_probs, eps)).sum(dim=-1).mean()


def view_loss_fn(teacher_logits, student_logits, teacher_temp, student_temp, centers, eps=1e-20):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim=-1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim=-1)
    return -(teacher_probs * log(student_probs, eps)).sum(dim=-1).mean()


class EsViTTrainer(nn.Module):

    def __init__(self, net, image_size, hidden_layer=-2, projection_hidden_size=256, num_classes_K=65336, projection_layers=4, student_temp=0.9, teacher_temp=0.04, local_upper_crop_scale=0.4, global_lower_crop_scale=0.5, moving_average_decay=0.9, center_moving_average_decay=0.9, augment_fn=None, augment_fn2=None):
        super().__init__()
        self.net = net
        DEFAULT_AUG = torch.nn.Sequential(RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3), T.RandomGrayscale(p=0.2), T.RandomHorizontalFlip(), RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2), T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)
        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale=(0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale=(global_lower_crop_scale, 1.0))
        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer=hidden_layer)
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)
        self.register_buffer('teacher_view_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_view_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('teacher_region_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_region_centers', torch.zeros(1, num_classes_K))
        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        device = get_module_device(net)
        self
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
        new_teacher_view_centers = self.teacher_centering_ema_updater.update_average(self.teacher_view_centers, self.last_teacher_view_centers)
        self.teacher_view_centers.copy_(new_teacher_view_centers)
        new_teacher_region_centers = self.teacher_centering_ema_updater.update_average(self.teacher_region_centers, self.last_teacher_region_centers)
        self.teacher_region_centers.copy_(new_teacher_region_centers)

    def forward(self, x, return_embedding=False, return_projection=True, student_temp=None, teacher_temp=None):
        if return_embedding:
            return self.student_encoder(x, return_projection=return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)
        local_image_one, local_image_two = self.local_crop(image_one), self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)
        student_view_proj_one, student_region_proj_one, student_latent_one = self.student_encoder(local_image_one)
        student_view_proj_two, student_region_proj_two, student_latent_two = self.student_encoder(local_image_two)
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_view_proj_one, teacher_region_proj_one, teacher_latent_one = teacher_encoder(global_image_one)
            teacher_view_proj_two, teacher_region_proj_two, teacher_latent_two = teacher_encoder(global_image_two)
        view_loss_fn_ = partial(view_loss_fn, student_temp=default(student_temp, self.student_temp), teacher_temp=default(teacher_temp, self.teacher_temp), centers=self.teacher_view_centers)
        region_loss_fn_ = partial(region_loss_fn, student_temp=default(student_temp, self.student_temp), teacher_temp=default(teacher_temp, self.teacher_temp), centers=self.teacher_region_centers)
        teacher_view_logits_avg = torch.cat((teacher_view_proj_one, teacher_view_proj_two)).mean(dim=0)
        self.last_teacher_view_centers.copy_(teacher_view_logits_avg)
        teacher_region_logits_avg = torch.cat((teacher_region_proj_one, teacher_region_proj_two)).mean(dim=(0, 1))
        self.last_teacher_region_centers.copy_(teacher_region_logits_avg)
        view_loss = (view_loss_fn_(teacher_view_proj_one, student_view_proj_two) + view_loss_fn_(teacher_view_proj_two, student_view_proj_one)) / 2
        region_loss = (region_loss_fn_(teacher_region_proj_one, student_region_proj_two, teacher_latent_one, student_latent_two) + region_loss_fn_(teacher_region_proj_two, student_region_proj_one, teacher_latent_two, student_latent_one)) / 2
        return (view_loss + region_loss) / 2


def apply_tuple_or_single(fn, val):
    if isinstance(val, tuple):
        return tuple(map(fn, val))
    return fn(val)


def clone_and_detach(t):
    return t.clone().detach()


def identity(t):
    return t


class Extractor(nn.Module):

    def __init__(self, vit, device=None, layer=None, layer_name='transformer', layer_save_input=False, return_embeddings_only=False, detach=True):
        super().__init__()
        self.vit = vit
        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device
        self.layer = layer
        self.layer_name = layer_name
        self.layer_save_input = layer_save_input
        self.return_embeddings_only = return_embeddings_only
        self.detach_fn = clone_and_detach if detach else identity

    def _hook(self, _, inputs, output):
        layer_output = inputs if self.layer_save_input else output
        self.latents = apply_tuple_or_single(self.detach_fn, layer_output)

    def _register_hook(self):
        if not exists(self.layer):
            assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
            layer = getattr(self.vit, self.layer_name)
        else:
            layer = self.layer
        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        del self.latents
        self.latents = None

    def forward(self, img, return_embeddings_only=False):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        pred = self.vit(img)
        target_device = self.device if exists(self.device) else img.device
        latents = apply_tuple_or_single(lambda t: t, self.latents)
        if return_embeddings_only or self.return_embeddings_only:
            return latents
        return pred, latents


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


class Adapter(nn.Module):

    def __init__(self, *, vit, num_memories_per_layer=10, num_classes=2):
        super().__init__()
        assert isinstance(vit, ViT)
        dim = vit.cls_token.shape[-1]
        layers = len(vit.transformer.layers)
        num_patches = vit.pos_embedding.shape[-2]
        self.vit = vit
        freeze_all_layers_(vit)
        self.memory_cls_token = nn.Parameter(torch.randn(dim))
        self.memories_per_layer = nn.Parameter(torch.randn(layers, num_memories_per_layer, dim))
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        attn_mask = torch.ones((num_patches, num_patches), dtype=torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value=False)
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value=True)
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img):
        b = img.shape[0]
        tokens = self.vit.img_to_tokens(img)
        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b=b)
        tokens = torch.cat((memory_cls_tokens, tokens), dim=1)
        out = self.vit.transformer(tokens, memories=self.memories_per_layer, attn_mask=self.attn_mask)
        memory_cls_tokens = out[:, 0]
        return self.mlp_head(memory_cls_tokens)


def always(val):
    return lambda *args, **kwargs: val


class LeViT(nn.Module):

    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_mult, stages=3, dim_key=32, dim_value=64, dropout=0.0, num_distill_classes=None):
        super().__init__()
        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)
        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'
        self.conv_embedding = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.Conv2d(128, dims[0], 3, stride=2, padding=1))
        fmap_size = image_size // 2 ** 4
        layers = []
        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == stages - 1
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))
            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out=next_dim, downsample=True))
                fmap_size = ceil(fmap_size / 2)
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), Rearrange('... () () -> ...'))
        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.conv_embedding(img)
        x = self.backbone(x)
        x = self.pool(x)
        out = self.mlp_head(x)
        distill = self.distill_head(x)
        if exists(distill):
            return out, distill
        return out


class ExcludeCLS(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim=1)


class LocalViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])


class MAE(nn.Module):

    def __init__(self, *, encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, decoder_heads=8, decoder_dim_head=64):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:num_patches + 1]
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


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


def MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1
    net = nn.Sequential(nn.Conv2d(dim_in, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim), nn.BatchNorm2d(hidden_dim), nn.GELU(), SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate), nn.Conv2d(hidden_dim, dim_out, 1), nn.BatchNorm2d(dim_out))
    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)
    return net


class MaxViT(nn.Module):

    def __init__(self, *, num_classes, dim, depth, dim_head=32, dim_conv_stem=None, window_size=7, mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, dropout=0.1, channels=3):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        dim_conv_stem = default(dim_conv_stem, dim)
        self.conv_stem = nn.Sequential(nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1), nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1))
        num_stages = len(depth)
        dims = tuple(map(lambda i: 2 ** i * dim, range(num_stages)))
        dims = dim_conv_stem, *dims
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList([])
        w = window_size
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                block = nn.Sequential(MBConv(stage_dim_in, layer_dim, downsample=is_first, expansion_rate=mbconv_expansion_rate, shrinkage_rate=mbconv_shrinkage_rate), Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w), PreNormResidual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)), PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)), Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'), Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w), PreNormResidual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)), PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)), Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'))
                self.layers.append(block)
        self.mlp_head = nn.Sequential(Reduce('b d h w -> b d', 'mean'), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes))

    def forward(self, x):
        x = self.conv_stem(x)
        for stage in self.layers:
            x = stage(x)
        return self.mlp_head(x)


class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.SiLU())


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.SiLU())


class MobileViTBlock(nn.Module):

    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2), depths=(2, 4, 3)):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        init_dim, *_, last_dim = channels
        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)
        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([MV2Block(channels[3], channels[4], 2, expansion), MobileViTBlock(dims[0], depths[0], channels[5], kernel_size, patch_size, int(dims[0] * 2))]))
        self.trunk.append(nn.ModuleList([MV2Block(channels[5], channels[6], 2, expansion), MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1] * 4))]))
        self.trunk.append(nn.ModuleList([MV2Block(channels[7], channels[8], 2, expansion), MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2] * 4))]))
        self.to_logits = nn.Sequential(conv_1x1_bn(channels[-2], last_dim), Reduce('b c h w -> b c', 'mean'), nn.Linear(channels[-1], num_classes, bias=False))

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.stem:
            x = conv(x)
        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)
        return self.to_logits(x)


class MPPLoss(nn.Module):

    def __init__(self, patch_size, channels, output_channel_bits, max_pixel_val, mean, std):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val
        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device
        bin_size = mpv / 2 ** bits
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean
        target = target.clamp(max=mpv)
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1=p, p2=p).contiguous()
        channel_bins = torch.arange(bin_size, mpv, bin_size, device=device)
        discretized_target = torch.bucketize(avg_target, channel_bins)
        bin_mask = (2 ** bits) ** torch.arange(0, c, device=device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')
        target_label = torch.sum(bin_mask * discretized_target, dim=-1)
        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)
    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob


class MPP(nn.Module):

    def __init__(self, transformer, patch_size, dim, output_channel_bits=3, channels=3, max_pixel_val=1.0, mask_prob=0.15, replace_prob=0.5, random_patch_prob=0.5, mean=None, std=None):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits, max_pixel_val, mean, std)
        self.to_bits = nn.Linear(dim, 2 ** (output_channel_bits * channels))
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):
        transformer = self.transformer
        img = input.clone().detach()
        p = self.patch_size
        input = rearrange(input, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        mask = get_mask_subset_with_prob(input, self.mask_prob)
        masked_input = input.clone().detach()
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input, random_patch_sampling_prob)
            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0, input.shape[1], (input.shape[0], input.shape[1]), device=input.device)
            randomized_input = masked_input[torch.arange(masked_input.shape[0]).unsqueeze(-1), random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[bool_random_patch_prob]
        replace_prob = prob_mask_like(input, self.replace_prob)
        bool_mask_replace = mask * replace_prob == True
        masked_input[bool_mask_replace] = self.mask_token
        masked_input = transformer.to_patch_embedding[-1](masked_input)
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)
        masked_input += transformer.pos_embedding[:, :n + 1]
        masked_input = transformer.dropout(masked_input)
        masked_input = transformer.transformer(masked_input, **kwargs)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]
        mpp_loss = self.loss(logits, img, mask)
        return mpp_loss


def Aggregate(dim, dim_out):
    return nn.Sequential(nn.Conv2d(dim, dim_out, 3, padding=1), LayerNorm(dim_out), nn.MaxPool2d(3, stride=2, padding=1))


class NesT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, heads, num_hierarchies, block_repeats, mlp_mult=4, channels=3, dim_head=64, dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)
        seq_len = (fmap_size // blocks) ** 2
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [(2 ** i) for i in reversed(hierarchies)]
        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]
        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size), nn.Conv2d(patch_dim, layer_dims[0], 1))
        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout), Aggregate(dim_in, dim_out) if not is_last else nn.Identity()]))
        self.mlp_head = nn.Sequential(LayerNorm(last_dim), Reduce('b c h w -> b c', 'mean'), nn.Linear(last_dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
        return self.mlp_head(x)


class Parallel(nn.Module):

    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])


class Pool(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]
        cls_token = self.cls_ff(cls_token)
        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h=int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        return torch.cat((cls_token, tokens), dim=1)


class PiT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.0, emb_dropout=0.0, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(nn.Unfold(kernel_size=patch_size, stride=patch_size // 2), Rearrange('b c n -> b n c'), nn.Linear(patch_dim, dim))
        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        layers = []
        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < len(depth) - 1
            layers.append(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))
            if not_last:
                layers.append(Pool(dim))
                dim *= 2
        self.layers = nn.Sequential(*layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.layers(x)
        return self.mlp_head(x[:, 0])


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Recorder(nn.Module):

    def __init__(self, vit, device=None):
        super().__init__()
        self.vit = vit
        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        pred = self.vit(img)
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t, self.recordings))
        attns = torch.stack(recordings, dim=1) if len(recordings) > 0 else None
        return pred, attns


class Downsample(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class PEG(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, stride=1))

    def forward(self, x):
        return self.proj(x)


class R2LTransformer(nn.Module):

    def __init__(self, dim, *, window_size, depth=4, heads=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.window_size = window_size
        rel_positions = 2 * window_size - 1
        self.local_rel_pos_bias = nn.Embedding(rel_positions ** 2, heads)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout), FeedForward(dim, dropout=ff_dropout)]))

    def forward(self, local_tokens, region_tokens):
        device = local_tokens.device
        lh, lw = local_tokens.shape[-2:]
        rh, rw = region_tokens.shape[-2:]
        window_size_h, window_size_w = lh // rh, lw // rw
        local_tokens = rearrange(local_tokens, 'b c h w -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b c h w -> b (h w) c')
        h_range = torch.arange(window_size_h, device=device)
        w_range = torch.arange(window_size_w, device=device)
        grid_x, grid_y = torch.meshgrid(h_range, w_range, indexing='ij')
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = grid[:, :, None] - grid[:, None, :] + (self.window_size - 1)
        bias_indices = (grid * torch.tensor([1, self.window_size * 2 - 1], device=device)[:, None, None]).sum(dim=0)
        rel_pos_bias = self.local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        rel_pos_bias = F.pad(rel_pos_bias, (1, 0, 1, 0), value=0)
        for attn, ff in self.layers:
            region_tokens = attn(region_tokens) + region_tokens
            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h=lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1=window_size_h, p2=window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')
            region_and_local_tokens = torch.cat((region_tokens, local_tokens), dim=1)
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias=rel_pos_bias) + region_and_local_tokens
            region_and_local_tokens = ff(region_and_local_tokens) + region_and_local_tokens
            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h=lh // window_size_h, w=lw // window_size_w, p1=window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n=rh * rw)
        local_tokens = rearrange(local_tokens, 'b (h w) c -> b c h w', h=lh, w=lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b c h w', h=rh, w=rw)
        return local_tokens, region_tokens


def divisible_by(val, d):
    return val % d == 0


class RegionViT(nn.Module):

    def __init__(self, *, dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), window_size=7, num_classes=1000, tokenize_local_3_conv=False, local_patch_size=4, use_peg=False, attn_dropout=0.0, ff_dropout=0.0, channels=3):
        super().__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'
        self.local_patch_size = local_patch_size
        region_patch_size = local_patch_size * window_size
        self.region_patch_size = local_patch_size * window_size
        init_dim, *_, last_dim = dim
        if tokenize_local_3_conv:
            self.local_encoder = nn.Sequential(nn.Conv2d(3, init_dim, 3, 2, 1), nn.LayerNorm(init_dim), nn.GELU(), nn.Conv2d(init_dim, init_dim, 3, 2, 1), nn.LayerNorm(init_dim), nn.GELU(), nn.Conv2d(init_dim, init_dim, 3, 1, 1))
        else:
            self.local_encoder = nn.Conv2d(3, init_dim, 8, 4, 3)
        self.region_encoder = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=region_patch_size, p2=region_patch_size), nn.Conv2d(region_patch_size ** 2 * channels, init_dim, 1))
        current_dim = init_dim
        self.layers = nn.ModuleList([])
        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and use_peg
            self.layers.append(nn.ModuleList([Downsample(current_dim, dim) if need_downsample else nn.Identity(), PEG(dim) if need_peg else nn.Identity(), R2LTransformer(dim, depth=num_layers, window_size=window_size, attn_dropout=attn_dropout, ff_dropout=ff_dropout)]))
            current_dim = dim
        self.to_logits = nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.LayerNorm(last_dim), nn.Linear(last_dim, num_classes))

    def forward(self, x):
        *_, h, w = x.shape
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'
        local_tokens = self.local_encoder(x)
        region_tokens = self.region_encoder(x)
        for down, peg, transformer in self.layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)
        return self.to_logits(region_tokens)


class SpatialConv(nn.Module):

    def __init__(self, dim_in, dim_out, kernel, bias=False):
        super().__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel, padding=kernel // 2, bias=False)
        self.cls_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, fmap_dims):
        cls_token, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (h w) d -> b d h w', **fmap_dims)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        cls_token = self.cls_proj(cls_token)
        return torch.cat((cls_token, x), dim=1)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RvT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, use_rotary=True, use_ds_conv=True, use_glu=True):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, image_size, dropout, use_rotary, use_ds_conv, use_glu)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        b, _, h, w, p = *img.shape, self.patch_size
        x = self.to_patch_embedding(img)
        n = x.shape[1]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        fmap_dims = {'h': h // p, 'w': w // p}
        x = self.transformer(x, fmap_dims=fmap_dims)
        return self.mlp_head(x[:, 0])


class ChanLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ScalableSelfAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_key=32, dim_value=32, dropout=0.0, reduction_factor=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, reduction_factor, stride=reduction_factor, bias=False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, reduction_factor, stride=reduction_factor, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(dim_value * heads, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        height, width, heads = *x.shape[-2:], self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=height, y=width)
        return self.to_out(out)


class InteractiveWindowedSelfAttention(nn.Module):

    def __init__(self, dim, window_size, heads=8, dim_key=32, dim_value=32, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.local_interactive_module = nn.Conv2d(dim_value * heads, dim_value * heads, 3, padding=1)
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(dim_value * heads, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        height, width, heads, wsz = *x.shape[-2:], self.heads, self.window_size
        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert height % wsz_h == 0 and width % wsz_w == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        local_out = self.local_interactive_module(v)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h=heads, w1=wsz_h, w2=wsz_w), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x=height // wsz_h, y=width // wsz_w, w1=wsz_h, w2=wsz_w)
        out = out + local_out
        return self.to_out(out)


class ScalableViT(nn.Module):

    def __init__(self, *, num_classes, dim, depth, heads, reduction_factor, window_size=None, iwsa_dim_key=32, iwsa_dim_value=32, ssa_dim_key=32, ssa_dim_value=32, ff_expansion_factor=4, channels=3, dropout=0.0):
        super().__init__()
        self.to_patches = nn.Conv2d(channels, dim, 7, stride=4, padding=3)
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        num_stages = len(depth)
        dims = tuple(map(lambda i: 2 ** i * dim, range(num_stages)))
        hyperparams_per_stage = [heads, ssa_dim_key, ssa_dim_value, reduction_factor, iwsa_dim_key, iwsa_dim_value, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))
        self.layers = nn.ModuleList([])
        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == num_stages - 1
            self.layers.append(nn.ModuleList([Transformer(dim=layer_dim, depth=layer_depth, heads=layer_heads, ff_expansion_factor=ff_expansion_factor, dropout=dropout, ssa_dim_key=layer_ssa_dim_key, ssa_dim_value=layer_ssa_dim_value, ssa_reduction_factor=layer_ssa_reduction_factor, iwsa_dim_key=layer_iwsa_dim_key, iwsa_dim_value=layer_iwsa_dim_value, iwsa_window_size=layer_window_size, norm_output=not is_last), Downsample(layer_dim, layer_dim * 2) if not is_last else None]))
        self.mlp_head = nn.Sequential(Reduce('b d h w -> b d', 'mean'), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes))

    def forward(self, img):
        x = self.to_patches(img)
        for transformer, downsample in self.layers:
            x = transformer(x)
            if exists(downsample):
                x = downsample(x)
        return self.mlp_head(x)


class OverlappingPatchEmbed(nn.Module):

    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class DSSA(nn.Module):

    def __init__(self, dim, heads=8, dim_head=32, dropout=0.0, window_size=7):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.window_tokens = nn.Parameter(torch.randn(dim))
        self.window_tokens_to_qk = nn.Sequential(nn.LayerNorm(dim_head), nn.GELU(), Rearrange('b h n c -> b (h c) n'), nn.Conv1d(inner_dim, inner_dim * 2, 1), Rearrange('b (h c) n -> b h n c', h=heads))
        self.window_attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """
        batch, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        assert height % wsz == 0 and width % wsz == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        num_windows = height // wsz * (width // wsz)
        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1=wsz, w2=wsz)
        w = repeat(self.window_tokens, 'c -> b c 1', b=x.shape[0])
        x = torch.cat((w, x), dim=-1)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        q = q * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]
        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x=height // wsz, y=width // wsz, w1=wsz, w2=wsz)
            return self.to_out(fmap)
        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x=height // wsz, y=width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // wsz, y=width // wsz)
        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim=-1)
        w_q = w_q * self.scale
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        w_attn = self.window_attend(w_dots)
        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)
        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x=height // wsz, y=width // wsz, w1=wsz, w2=wsz)
        return self.to_out(fmap)


class SepViT(nn.Module):

    def __init__(self, *, num_classes, dim, depth, heads, window_size=7, dim_head=32, ff_mult=4, channels=3, dropout=0.0):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        num_stages = len(depth)
        dims = tuple(map(lambda i: 2 ** i * dim, range(num_stages)))
        dims = channels, *dims
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        strides = 4, *((2,) * (num_stages - 1))
        hyperparams_per_stage = [heads, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))
        self.layers = nn.ModuleList([])
        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            is_last = ind == num_stages - 1
            self.layers.append(nn.ModuleList([OverlappingPatchEmbed(layer_dim_in, layer_dim, stride=layer_stride), PEG(layer_dim), Transformer(dim=layer_dim, depth=layer_depth, heads=layer_heads, ff_mult=ff_mult, dropout=dropout, norm_output=not is_last)]))
        self.mlp_head = nn.Sequential(Reduce('b d h w -> b d', 'mean'), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes))

    def forward(self, x):
        for ope, peg, transformer in self.layers:
            x = ope(x)
            x = peg(x)
            x = transformer(x)
        return self.mlp_head(x)


class SimMIM(nn.Module):

    def __init__(self, *, encoder, masking_ratio=0.5):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        batch_range = torch.arange(batch, device=device)[:, None]
        pos_emb = self.encoder.pos_embedding[:, 1:num_patches + 1]
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_patches)
        mask_tokens = mask_tokens + pos_emb
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device=device).topk(k=num_masked, dim=-1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device=device).scatter_(-1, masked_indices, 1).bool()
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)
        encoded = self.encoder.transformer(tokens)
        encoded_mask_tokens = encoded[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(encoded_mask_tokens)
        masked_patches = patches[batch_range, masked_indices]
        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss


class PatchDropout(nn.Module):

    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        b, n, _, device = *x.shape, x.device
        batch_indices = torch.arange(b, device=device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device=device).topk(num_patches_keep, dim=-1).indices
        return x[batch_indices, patch_indices_keep]


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert dim % 4 == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / temperature ** omega
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class SimpleViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, patch_dropout=0.5):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = image_height // patch_height * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width), nn.Linear(patch_dim, dim))
        self.patch_dropout = PatchDropout(patch_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.patch_dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)


class PatchEmbedding(nn.Module):

    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.proj = nn.Conv2d(patch_size ** 2 * dim, dim_out, 1)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=p, p2=p)
        return self.proj(fmap)


class LocalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, patch_size=7):
        super().__init__()
        inner_dim = dim_head * heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))
        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1=p, p2=p)
        q, k, v = self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h=h), (q, k, v))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h=h, x=x, y=y, p1=p, p2=p)
        return self.to_out(out)


class GlobalAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, k=7):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride=k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        return self.to_out(out)


class TwinsSVT(nn.Module):

    def __init__(self, *, num_classes, s1_emb_dim=64, s1_patch_size=4, s1_local_patch_size=7, s1_global_k=7, s1_depth=1, s2_emb_dim=128, s2_patch_size=2, s2_local_patch_size=7, s2_global_k=7, s2_depth=1, s3_emb_dim=256, s3_patch_size=2, s3_local_patch_size=7, s3_global_k=7, s3_depth=5, s4_emb_dim=512, s4_patch_size=2, s4_local_patch_size=7, s4_global_k=7, s4_depth=4, peg_kernel_size=3, dropout=0.0):
        super().__init__()
        kwargs = dict(locals())
        dim = 3
        layers = []
        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'
            dim_next = config['emb_dim']
            layers.append(nn.Sequential(PatchEmbedding(dim=dim, dim_out=dim_next, patch_size=config['patch_size']), Transformer(dim=dim_next, depth=1, local_patch_size=config['local_patch_size'], global_k=config['global_k'], dropout=dropout, has_local=not is_last), PEG(dim=dim_next, kernel_size=peg_kernel_size), Transformer(dim=dim_next, depth=config['depth'], local_patch_size=config['local_patch_size'], global_k=config['global_k'], dropout=dropout, has_local=not is_last)))
            dim = dim_next
        self.layers = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), Rearrange('... () () -> ...'), nn.Linear(dim, num_classes))

    def forward(self, x):
        return self.layers(x)


class LSA(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SPT(nn.Module):

    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim))

    def forward(self, x):
        shifts = (1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1)
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class PatchMerger(nn.Module):

    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        return torch.matmul(attn, x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChanLayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEmbedLayer,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DepthWiseConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dropsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExcludeCLS,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale,
     lambda: ([], {'dim': 4, 'fn': _mock_layer(), 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MBConvResidual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MV2Block,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OverlappingPatchEmbed,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PEG,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Parallel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchDropout,
     lambda: ([], {'prob': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchMerger,
     lambda: ([], {'dim': 4, 'num_tokens_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreNormResidual,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProjectInOut,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RandomApply,
     lambda: ([], {'fn': _mock_layer(), 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_vit_pytorch(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

