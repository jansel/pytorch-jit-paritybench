import sys
_module = sys.modules[__name__]
del sys
convert_coco_object = _module
convert_yfcc14m = _module
create_subset = _module
process_redcaps = _module
datasets = _module
builder = _module
formatting = _module
imagenet_template = _module
tokenizer = _module
demo_seg = _module
main_group_vit = _module
main_seg = _module
models = _module
group_vit = _module
misc = _module
multi_label_contrastive = _module
transformer = _module
utils = _module
custom_import = _module
coco = _module
pascal_context = _module
pascal_voc12 = _module
coco_object = _module
pascal_voc = _module
evaluation = _module
group_vit_seg = _module
checkpoint = _module
config = _module
logger = _module
lr_scheduler = _module
misc = _module
optimizer = _module

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


import random


import warnings


from functools import partial


import numpy as np


import torch


import torch.distributed as dist


from torchvision import transforms


import time


from collections import defaultdict


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.checkpoint as checkpoint


import math


from torch import nn


import matplotlib.pyplot as plt


import collections.abc


from torch import optim as optim


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def gumbel_softmax(logits: torch.Tensor, tau: float=1, hard: bool=False, dim: int=-1) ->torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0, device=logits.device, dtype=logits.dtype), torch.tensor(1.0, device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, hard=True, gumbel=False, gumbel_tau=1.0, sum_assign=False, assign_eps=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):
        if gumbel is None:
            gumbel = self.gumbel
        if hard is None:
            hard = self.hard
        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        elif hard:
            attn = hard_softmax(attn, dim=attn_dim)
        else:
            attn = F.softmax(attn, dim=attn_dim)
        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        raw_attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None
        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \nhard: {self.hard}, \ngumbel: {self.gumbel}, \nsum_assign={self.sum_assign}, \ngumbel_tau: {self.gumbel_tau}, \nassign_eps: {self.assign_eps}'


class Attention(nn.Module):

    def __init__(self, dim, num_heads, out_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \nqkv_bias={self.scale}, \nqkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self, *, dim, out_dim, num_heads, num_group_token, num_output_group, norm_layer, mlp_ratio=(0.5, 4.0), hard=True, gumbel=True, sum_assign=False, assign_eps=1.0, gumbel_tau=1.0):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)
        self.assign = AssignAttention(dim=dim, num_heads=1, qkv_bias=True, hard=hard, gumbel=gumbel, gumbel_tau=gumbel_tau, sum_assign=sum_assign, assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \ngumbel={self.gumbel}, \nsum_assign={self.sum_assign}, \nnum_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens
        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))
        return new_x, attn_dict


class AttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self, dim, num_input_token, depth, num_heads, num_group_token, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, group_projector=None, zero_init_group_token=False):
        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=0.02)
        else:
            self.group_token = None
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(AttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint
        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \ninput_resolution={self.input_resolution}, \ndepth={self.depth}, \nnum_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None
        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)
        x, group_token = self.split_x(cat_x)
        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)
        return x, group_token, attn_dict


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1), int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class Result:

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W
    N = pos_embed.shape[1]
    if num_patches == N and W == H:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), size=(H, W), mode='bicubic', align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


class GroupViT(nn.Module):
    """ Group Vision Transformer
        A PyTorch impl of : `GroupViT: Semantic Segmentation Emerges from Text Supervision`  -
          https://arxiv.org/pdf/2202.11094.pdf

    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        patch_size (int | tuple[int]): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 0
        embed_dim (int): Patch embedding dimension. Default: 384
        embed_factors (list[int]): Embedding dim multipliers for each stage.
        depths (list[int]): Depth of each stage
        num_heads (list[int]): Number of heads for each stage
        num_group_tokens (list[int]): Number of group tokens for each stage
        num_output_group (list[int]): Number of output groups for each stage
        hard_assignment (bool): Whether to use hard assignment or not. Default: True
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pos_embed_type (str): Type of positional embedding. Default: 'simple'
        freeze_patch_embed (bool): Whether to freeze patch embedding. Default: False
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=384, embed_factors=[1, 1, 1], depths=[6, 3, 3], num_heads=[6, 6, 6], num_group_tokens=[64, 8, 0], num_output_groups=[64, 8], hard_assignment=True, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True, use_checkpoint=False, pos_embed_type='simple', freeze_patch_embed=False):
        super().__init__()
        assert patch_size in [4, 8, 16]
        self.num_classes = num_classes
        assert len(embed_factors) == len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        assert len(depths) - 1 == len(num_output_groups)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * embed_factors[len(depths) - 1])
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']
        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(img_size=img_size, kernel_size=patch_size, stride=patch_size, padding=0, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError
        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_input_token = num_patches
        num_output_token = num_input_token
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = embed_dim * embed_factors[i_layer + 1]
                downsample = GroupingBlock(dim=dim, out_dim=out_dim, num_heads=num_heads[i_layer], num_group_token=num_group_tokens[i_layer], num_output_group=num_output_groups[i_layer], norm_layer=norm_layer, hard=hard_assignment, gumbel=hard_assignment)
                num_output_token = num_output_groups[i_layer]
            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(norm_layer(prev_dim), MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))
                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, norm_layer(prev_dim), nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None
            layer = GroupingLayer(dim=dim, num_input_token=num_input_token, depth=depths[i_layer], num_heads=num_heads[i_layer], num_group_token=num_group_tokens[i_layer], mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint, group_projector=group_projector, zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool=True):
        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']
            pos_embed = self.pos_embed
            if load_pos_embed.shape != pos_embed.shape:
                H_new = int(self.patch_embed.num_patches ** 0.5)
                W_new = H_new
                H_ori = int(load_pos_embed.shape[1] ** 0.5)
                W_ori = H_ori
                load_pos_embed = F.interpolate(rearrange(load_pos_embed, 'b (h w) c -> b c h w', h=H_ori, w=W_ori, b=1), size=(H_new, W_new), mode='bicubic', align_corners=False)
                load_pos_embed = rearrange(load_pos_embed, 'b c h w -> b (h w) c', h=H_new, w=W_new)
                state_dict['pos_embed'] = load_pos_embed
        return super().load_state_dict(state_dict, strict)

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed

    def forward_features(self, x, *, return_attn=False):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)
        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)
        group_token = None
        attn_dict_list = []
        for layer in self.layers:
            x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)
            attn_dict_list.append(attn_dict)
        x = self.norm(x)
        return x, group_token, attn_dict_list

    def forward_image_head(self, x):
        """

        Args:
            x: shape [B, L, C]

        Returns:

        """
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False):
        x, group_token, attn_dicts = self.forward_features(x, return_attn=return_attn)
        x_feat = x if return_feat else None
        outs = Result(as_dict=as_dict)
        outs.append(self.forward_image_head(x), name='x')
        if return_feat:
            outs.append(x_feat, name='feat')
        if return_attn:
            outs.append(attn_dicts, name='attn_dicts')
        return outs.as_return()


class ProjectMLP(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)
        self.linear_out = nn.Conv1d(in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            add_dim = True
        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')
        if add_dim:
            x = x.squeeze(1)
        return x


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class MultiLabelContrastive(nn.Module):

    def __init__(self, img_encoder, text_encoder, output_dim=256, contrast_temperature=0.07, proj_num_layers=2, multi_label=0, share_temperature=False, multi_label_loss_weight=1.0):
        super().__init__()
        self.img_encoder = MODELS.build(img_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        self.contrast_temperature = contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.soft_cross_entropy = SoftTargetCrossEntropy()
        self.proj_num_layers = proj_num_layers
        self.multi_label = multi_label
        if proj_num_layers > 0:
            self.img_projector = ProjectMLP(in_dim=self.img_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.text_projector = ProjectMLP(in_dim=self.text_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.img_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_projector)
            self.text_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_projector)
        else:
            self.img_projector = nn.Identity()
            self.text_projector = nn.Identity()
        self.share_temperature = share_temperature
        if self.with_multi_label and not self.share_temperature:
            self.multi_label_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.multi_label_loss_weight = multi_label_loss_weight

    @property
    def with_multi_label(self):
        return self.multi_label > 0

    def loss(self, image_x, text_x):
        batch_size = image_x.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)
        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)
        loss = 0.5 * (loss_img + loss_text)
        return loss

    def multi_label_loss(self, image_feat, text_feat):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')
        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)
        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]
        pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
        pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')
        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')
        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        labels_per_img = F.one_hot(torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * dist.get_rank(), num_classes=dist.get_world_size())
        labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        labels_per_text = F.one_hot(torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * dist.get_rank(), num_classes=dist.get_world_size())
        labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')
        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, labels_per_text)
        loss = 0.5 * (loss_img + loss_text)
        return loss

    def encode_image(self, image, *, return_feat=False, as_dict=False):
        outs = Result(as_dict)
        img_outs = self.img_encoder(image, return_feat=return_feat, as_dict=True)
        outs.append(self.img_projector(img_outs['x']), 'image_x')
        if return_feat:
            outs.append(self.img_projector(img_outs['feat']), 'image_feat')
        return outs.as_return()

    def encode_text(self, text, *, as_dict=False):
        assert text.ndim in [2, 3], text.ndim
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True
        outs = Result(as_dict=as_dict)
        x = self.text_encoder(text)
        text_x = self.text_projector(x)
        outs.append(text_x, 'text_x')
        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            text_multi_label_x = text_x[:, 1:]
            text_x = text_x[:, 0]
            outs.update(text_x=text_x, text_multi_label_x=text_multi_label_x)
        return outs.as_return()

    def forward_train(self, image, text):
        image_outs = self.encode_image(image, as_dict=True)
        image_x = image_outs['image_x']
        text_outs = self.encode_text(text, as_dict=True)
        text_x = text_outs['text_x']
        losses = self.loss(image_x, text_x)
        losses_dict = dict(loss=losses)
        if self.with_multi_label:
            image_multi_label_x = image_x.unsqueeze(1)
            text_multi_label_x = text_outs['text_multi_label_x']
            losses_dict['multi_label_loss'] = self.multi_label_loss(image_multi_label_x, text_multi_label_x) * self.multi_label_loss_weight
        return losses_dict

    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)

    def forward(self, image, text):
        if self.training:
            return self.forward_train(image, text)
        else:
            return self.forward_test(image, text)

    @torch.no_grad()
    def build_text_embedding(self, text):
        """

        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

        Returns:

        """
        text = text
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        text_tokens = self.encode_text(text)
        text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
        text_tokens = text_tokens.mean(dim=1)
        text_tokens = F.normalize(text_tokens, dim=-1)
        return text_tokens

    @torch.no_grad()
    def zero_shot_pred(self, image, text):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logits_per_image = image_features @ text.t()
        return logits_per_image


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None, use_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        proj_std = self.width ** -0.5 * (2 * self.layers) ** -0.5
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        for resblock in self.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(resblock, x)
            else:
                x = resblock(x)
        return x


class TextTransformer(nn.Module):

    def __init__(self, context_length: int, width: int, layers: int, vocab_size, use_checkpoint=False):
        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = Transformer(width=width, layers=layers, heads=heads, attn_mask=self.build_attention_mask(), use_checkpoint=use_checkpoint)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def forward(self, text, *, as_dict=False):
        x = self.token_embedding(text)
        outs = Result(as_dict=as_dict)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        outs.append(x, name='x')
        return outs.as_return()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MixerMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_NVlabs_GroupViT(_paritybench_base):
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

