import sys
_module = sys.modules[__name__]
del sys
dinat = _module
dinats = _module
extras = _module
isotropic = _module
nat = _module
train = _module
validate = _module
coco_detection = _module
coco_instance = _module
coco_instance_semantic = _module
default_runtime = _module
cascade_mask_rcnn_dinat = _module
cascade_mask_rcnn_dinats = _module
cascade_mask_rcnn_nat = _module
mask_rcnn_dinat = _module
mask_rcnn_dinats = _module
mask_rcnn_nat = _module
schedule_3x = _module
cascade_mask_rcnn_dinat_base_3x_coco = _module
cascade_mask_rcnn_dinat_large_3x_coco = _module
cascade_mask_rcnn_dinat_mini_3x_coco = _module
cascade_mask_rcnn_dinat_small_3x_coco = _module
cascade_mask_rcnn_dinat_tiny_3x_coco = _module
mask_rcnn_dinat_mini_3x_coco = _module
mask_rcnn_dinat_small_3x_coco = _module
mask_rcnn_dinat_tiny_3x_coco = _module
cascade_mask_rcnn_dinat_s_base_3x_coco = _module
cascade_mask_rcnn_dinat_s_large_3x_coco = _module
cascade_mask_rcnn_dinat_s_small_3x_coco = _module
cascade_mask_rcnn_dinat_s_tiny_3x_coco = _module
mask_rcnn_dinat_s_small_3x_coco = _module
mask_rcnn_dinat_s_tiny_3x_coco = _module
cascade_mask_rcnn_nat_base_3x_coco = _module
cascade_mask_rcnn_nat_mini_3x_coco = _module
cascade_mask_rcnn_nat_small_3x_coco = _module
cascade_mask_rcnn_nat_tiny_3x_coco = _module
mask_rcnn_nat_mini_3x_coco = _module
mask_rcnn_nat_small_3x_coco = _module
mask_rcnn_nat_tiny_3x_coco = _module
dinat = _module
dinats = _module
extras = _module
get_flops = _module
nat = _module
test = _module
train = _module
dinat = _module
train_m2f = _module
ade20k = _module
upernet_dinat = _module
upernet_dinats = _module
upernet_nat = _module
schedule_160k = _module
upernet_dinat_base_512x512_160k_ade20k = _module
upernet_dinat_large_640x640_160k_ade20k = _module
upernet_dinat_mini_512x512_160k_ade20k = _module
upernet_dinat_small_512x512_160k_ade20k = _module
upernet_dinat_tiny_512x512_160k_ade20k = _module
upernet_dinat_s_base_512x512_160k_ade20k = _module
upernet_dinat_s_large_640x640_160k_ade20k = _module
upernet_dinat_s_small_512x512_160k_ade20k = _module
upernet_dinat_s_tiny_512x512_160k_ade20k = _module
upernet_nat_base_512x512_160k_ade20k = _module
upernet_nat_mini_512x512_160k_ade20k = _module
upernet_nat_small_512x512_160k_ade20k = _module
upernet_nat_tiny_512x512_160k_ade20k = _module
dinat = _module
dinats = _module
extras = _module
get_flops = _module
nat = _module
test = _module
train = _module
gen_salient_maps = _module

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


import torch.nn as nn


from torch.nn.functional import pad


import time


import logging


from collections import OrderedDict


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


import torch.nn.parallel


import numpy as np


import warnings


import copy


from torchvision import transforms as T


from torchvision.transforms.functional import to_pil_image


import random


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


class NATransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, kernel_size=7, dilation=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilations=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([NATransformerLayer(dim=dim, num_heads=num_heads, kernel_size=kernel_size, dilation=1 if dilations is None else dilations[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, x


class PatchEmbed(nn.Module):
    """
    From Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = None if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DiNAT_s(nn.Module):

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], kernel_size=7, dilations=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, out_indices=(0, 1, 2, 3), pretrained=None, frozen_stages=-1):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], kernel_size=kernel_size, dilations=None if dilations is None else dilations[i_layer], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()
        if pretrained is not None:
            self.init_weights(pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()


class MHSARPB(nn.Module):
    """
    Self Attention + RPB
    """

    def __init__(self, dim, input_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.input_size = input_size[0] if type(input_size) is tuple else input_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, 2 * self.input_size - 1, 2 * self.input_size - 1))
        trunc_normal_(self.rpb, std=0.02)
        coords_h = torch.arange(self.input_size)
        coords_w = torch.arange(self.input_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.input_size - 1
        relative_coords[:, :, 1] += self.input_size - 1
        relative_coords[:, :, 0] *= 2 * self.input_size - 1
        relative_position_index = torch.flipud(torch.fliplr(relative_coords.sum(-1)))
        self.register_buffer('relative_position_index', relative_position_index)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_pb(self, attn):
        relative_position_bias = self.rpb.permute(1, 2, 0).flatten(0, 1)[self.relative_position_index.view(-1)].view(self.input_size ** 2, self.input_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return attn + relative_position_bias

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.input_size ** 2)
        if N != num_tokens:
            raise RuntimeError(f'Feature map size ({H} x {W}) is not equal to ' + f'expected size ({self.input_size} x {self.input_size}). ')
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.apply_pb(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        return self.proj_drop(self.proj(x))


class VisionTransformerLayer(nn.Module):

    def __init__(self, dim, input_size, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = MHSARPB(dim, input_size=input_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NATIsotropic(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, kernel_size=7, dilation=2, layer=NATransformerLayer, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.feature_map_size = img_size // patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.Sequential(*[layer(dim=embed_dim, input_size=self.feature_map_size, num_heads=num_heads, kernel_size=kernel_size, dilation=1 if i % 2 == 0 else dilation, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, self.feature_map_size, self.feature_map_size, self.embed_dim)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x.flatten(1, 2))
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvTokenizer(nn.Module):

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class NATLayer(nn.Module):

    def __init__(self, dim, num_heads, kernel_size=7, dilation=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):

    def __init__(self, dim, depth, num_heads, kernel_size, dilations=None, downsample=True, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([NATLayer(dim=dim, num_heads=num_heads, kernel_size=kernel_size, dilation=None if dilations is None else dilations[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, layer_scale=layer_scale) for i in range(depth)])
        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class NAT(nn.Module):

    def __init__(self, embed_dim, mlp_ratio, depths, num_heads, drop_path_rate=0.2, in_chans=3, kernel_size=7, dilations=None, out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, frozen_stages=-1, pretrained=None, layer_scale=None, **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio
        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i), depth=depths[i], num_heads=num_heads[i], kernel_size=kernel_size, dilations=None if dilations is None else dilations[i], mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, downsample=i < self.num_levels - 1, layer_scale=layer_scale)
            self.levels.append(level)
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(NAT, self).train(mode)
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)


class DiNAT(NAT):
    """
    DiNAT is NAT with dilations.
    It's that simple!
    """
    pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvDownsampler,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvTokenizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SHI_Labs_Neighborhood_Attention_Transformer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

