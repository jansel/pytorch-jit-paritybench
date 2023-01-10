import sys
_module = sys.modules[__name__]
del sys
dall_e = _module
decoder = _module
encoder = _module
utils = _module
coco_instance = _module
default_runtime = _module
cascade_mask_rcnn_r50_fpn = _module
cascade_mask_rcnn_swin_fpn = _module
cascade_mask_rcnn_vit_fpn = _module
mask_rcnn_r50_fpn = _module
mask_rcnn_vit_fpn = _module
schedule_1x = _module
mmcv_custom = _module
checkpoint = _module
layer_decay_optimizer_constructor = _module
prepare_rpe = _module
register_backbone = _module
runner = _module
checkpoint = _module
epoch_based_runner = _module
test = _module
train = _module
loader = _module
models = _module
head = _module
swin_transformer = _module
vision_transformer = _module
utils = _module
beit = _module
beit_fapn = _module
cae = _module
fapn = _module
mae = _module
ade20k = _module
ade20k_640x640 = _module
chase_db1 = _module
cityscapes = _module
cityscapes_769x769 = _module
drive = _module
hrf = _module
pascal_context = _module
pascal_voc12 = _module
pascal_voc12_aug = _module
stare = _module
cgnet = _module
fast_scnn = _module
fcn_hr18 = _module
fpn_r50 = _module
ocrnet_hr18 = _module
pointrend_r50 = _module
upernet_cae = _module
upernet_r50 = _module
schedule_160k = _module
schedule_20k = _module
schedule_320k = _module
schedule_40k = _module
schedule_80k = _module
apex_runner = _module
apex_iter_based_runner = _module
checkpoint = _module
optimizer = _module
checkpoint = _module
checkpoint_beit = _module
layer_decay_optimizer_constructor = _module
resize_transform = _module
train_api = _module
mmseg = _module
apis = _module
inference = _module
test = _module
train = _module
core = _module
evaluation = _module
class_names = _module
eval_hooks = _module
metrics = _module
seg = _module
builder = _module
sampler = _module
base_pixel_sampler = _module
ohem_pixel_sampler = _module
misc = _module
datasets = _module
ade = _module
builder = _module
coco_stuff = _module
custom = _module
dataset_wrappers = _module
pipelines = _module
compose = _module
formating = _module
loading = _module
test_time_aug = _module
transforms = _module
voc = _module
backbones = _module
cgnet = _module
fast_scnn = _module
hrnet = _module
mobilenet_v2 = _module
mobilenet_v3 = _module
resnest = _module
resnet = _module
resnext = _module
unet = _module
builder = _module
decode_heads = _module
ann_head = _module
apc_head = _module
aspp_head = _module
cascade_decode_head = _module
cc_head = _module
da_head = _module
decode_head = _module
dm_head = _module
dnl_head = _module
ema_head = _module
enc_head = _module
fcn_head = _module
fpn_head = _module
gc_head = _module
lraspp_head = _module
nl_head = _module
ocr_head = _module
point_head = _module
psa_head = _module
psp_head = _module
sep_aspp_head = _module
sep_fcn_head = _module
uper_head = _module
losses = _module
accuracy = _module
cross_entropy_loss = _module
lovasz_loss = _module
utils = _module
necks = _module
fpn = _module
segmentors = _module
base = _module
cascade_encoder_decoder = _module
encoder_decoder = _module
inverted_residual = _module
make_divisible = _module
res_layer = _module
se_layer = _module
self_attention_block = _module
up_conv_block = _module
ops = _module
encoding = _module
wrappers = _module
collect_env = _module
logger = _module
version = _module
test = _module
train = _module
dataset_folder = _module
datasets = _module
engine_for_finetuning = _module
engine_for_pretraining = _module
masking_generator = _module
optim_factory = _module
transforms = _module
utils = _module
crop = _module
engine_finetune = _module
lars = _module
lr_decay = _module
lr_sched = _module
misc = _module
pos_embed = _module
modeling_cae = _module
modeling_cae_helper = _module
modeling_discrete_vae = _module
modeling_finetune = _module
run_attentive = _module
run_class_finetuning = _module
run_linear = _module
run_pretraining = _module

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


import numpy as np


import torch.nn.functional as F


from collections import OrderedDict


from functools import partial


import math


import time


import warnings


import torchvision


from torch.optim import Optimizer


from torch.nn import functional as F


from scipy import interpolate


import torch.utils.checkpoint as checkpoint


import copy


import logging


import torch.distributed as dist


from math import sqrt


import random


from collections import defaultdict


from collections import deque


from torch import nn


from torch.utils import model_zoo


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from functools import reduce


from torch.utils.data import Dataset


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


import torch.utils.checkpoint as cp


from torch.nn.modules.batchnorm import _BatchNorm


from abc import ABCMeta


from abc import abstractmethod


import functools


from torch import nn as nn


from torch.utils import checkpoint as cp


from torchvision import datasets


from torchvision import transforms


from typing import Iterable


from typing import Optional


from torch import optim as optim


import torchvision.transforms.functional as F


from torch._six import inf


from torch.nn.modules.batchnorm import _NormBase


from torchvision.transforms import functional as F


from torch import einsum


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import torchvision.datasets as datasets


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.patch_shape = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CSyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, *args, with_var=False, **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x


class PSyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, *args, bunch_size, **kwargs):
        procs_per_bunch = min(bunch_size, utils.get_world_size())
        assert utils.get_world_size() % procs_per_bunch == 0
        n_bunch = utils.get_world_size() // procs_per_bunch
        ranks = list(range(utils.get_world_size()))
        None
        rank_groups = [ranks[i * procs_per_bunch:(i + 1) * procs_per_bunch] for i in range(n_bunch)]
        None
        process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
        bunch_id = utils.get_rank() // procs_per_bunch
        process_group = process_groups[bunch_id]
        None
        super(PSyncBatchNorm, self).__init__(*args, process_group=process_group, **kwargs)


class CustomSequential(nn.Sequential):
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DINOHead(nn.Module):

    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None
        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm = PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, 'unknown norm type {}'.format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, 'unknown act type {}'.format(act)
        return act


class iBOTHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, shared_head=False, **kwargs):
        super(iBOTHead, self).__init__(*args, norm=norm, act=act, last_norm=last_norm, nlayers=nlayers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, norm_last_layer=norm_last_layer, **kwargs)
        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None
            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None
            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)
        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        return x1, x2


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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn_out = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_out

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

    @staticmethod
    def compute_macs(module, input, output):
        B, N, C = input[0].shape
        module.__flops__ += module.flops(N) * B


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) ->str:
        return 'p={}'.format(self.drop_prob)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        self.attn_mask_dict = {}

    def create_attn_mask(self, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, L, C = x.shape
        H = int(sqrt(L))
        W = H
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if H is self.attn_mask_dict.keys():
                attn_mask = self.attn_mask_dict[H]
            else:
                self.attn_mask_dict[H] = self.create_attn_mask(H, W)
                attn_mask = self.attn_mask_dict[H]
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows, attn = self.attn(x_windows, attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size} mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H = int(sqrt(L))
        W = H
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x, _ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward_with_features(self, x):
        fea = []
        for blk in self.blocks:
            x, _ = blk(x)
            fea.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, fea

    def forward_with_attention(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, attns

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input channels.
        num_classes (int): Number of classes for classification head.
        embed_dim (int): Embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layers.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), ape=False, patch_norm=True, return_all_tokens=False, use_mean_pooling=True, masked_im_modeling=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.return_all_tokens = return_all_tokens
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, return_all_tokens=None, mask=None):
        x = self.patch_embed(x)
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x_region = self.norm(x)
        x = self.avgpool(x_region.transpose(1, 2))
        x = torch.flatten(x, 1)
        return_all_tokens = self.return_all_tokens if return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return torch.cat([x.unsqueeze(1), x_region], dim=1)
        return x

    def get_selfattention(self, x, n=1):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if n == 1:
            return self.get_last_selfattention(x)
        else:
            return self.get_all_selfattention(x)

    def get_last_selfattention(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
            else:
                x, attns = layer.forward_with_attention(x)
                return attns[-1]

    def get_all_selfattention(self, x):
        attn_out = []
        for layer in self.layers:
            x, attns = layer.forward_with_attention(x)
            attn_out += attns
        return attn_out

    def get_intermediate_layers(self, x, n=1, return_patch_avgpool=False):
        num_blks = sum(self.depths)
        start_idx = num_blks - n
        sum_cur = 0
        for i, d in enumerate(self.depths):
            sum_cur_new = sum_cur + d
            if start_idx >= sum_cur and start_idx < sum_cur_new:
                start_stage = i
                start_blk = start_idx - sum_cur
            sum_cur = sum_cur_new
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        output = []
        s = 0
        for i, layer in enumerate(self.layers):
            x, fea = layer.forward_with_features(x)
            if i >= start_stage:
                for x_ in fea[start_blk:]:
                    if i == len(self.layers) - 1:
                        x_ = self.norm(x_)
                    x_avg = torch.flatten(self.avgpool(x_.transpose(1, 2)), 1)
                    if return_patch_avgpool:
                        x_o = x_avg
                    else:
                        x_o = torch.cat((x_avg.unsqueeze(1), x_), dim=1)
                    output.append(x_o)
                start_blk = 0
        return output

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            if dist.get_rank() == 0:
                None
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = k.split('.')[0] in pretrained_layers or pretrained_layers[0] is '*' or 'relative_position_index' not in k or 'attn_mask' not in k
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'relative_position_bias_table' in k and v.size() != model_dict[k].size():
                        relative_position_bias_table_pretrained = v
                        relative_position_bias_table_current = model_dict[k]
                        L1, nH1 = relative_position_bias_table_pretrained.size()
                        L2, nH2 = relative_position_bias_table_current.size()
                        if nH1 != nH2:
                            logging.info(f'Error in loading {k}, passing')
                        elif L1 != L2:
                            logging.info('=> load_pretrained: resized variant: {} to {}'.format((L1, nH1), (L2, nH2)))
                            S1 = int(L1 ** 0.5)
                            S2 = int(L2 ** 0.5)
                            relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                            v = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                    if 'absolute_pos_embed' in k and v.size() != model_dict[k].size():
                        absolute_pos_embed_pretrained = v
                        absolute_pos_embed_current = model_dict[k]
                        _, L1, C1 = absolute_pos_embed_pretrained.size()
                        _, L2, C2 = absolute_pos_embed_current.size()
                        if C1 != C1:
                            logging.info(f'Error in loading {k}, passing')
                        elif L1 != L2:
                            logging.info('=> load_pretrained: resized variant: {} to {}'.format((1, L1, C1), (1, L2, C2)))
                            S1 = int(L1 ** 0.5)
                            S2 = int(L2 ** 0.5)
                            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                            absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                            v = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1).flatten(1, 2)
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def freeze_pretrained_layers(self, frozen_layers=[]):
        for name, module in self.named_modules():
            if name.split('.')[0] in frozen_layers or '.'.join(name.split('.')[0:2]) in frozen_layers or len(frozen_layers) > 0 and frozen_layers[0] is '*':
                for _name, param in module.named_parameters():
                    param.requires_grad = False
                logging.info('=> set param {} requires grad to False'.format(name))
        for name, param in self.named_parameters():
            if name.split('.')[0] in frozen_layers or (len(frozen_layers) > 0 and frozen_layers[0] is '*') and param.requires_grad is True:
                param.requires_grad = False
                logging.info('=> set param {} requires grad to False'.format(name))
        return self

    def get_num_layers(self):
        return sum(self.depths)

    def mask_model(self, x, mask):
        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        return x


class LP_BatchNorm(_NormBase):
    """ A variant used in linear probing.
    To freeze parameters (normalization operator specifically), model set to eval mode during linear probing.
    According to paper, an extra BN is used on the top of encoder to calibrate the feature magnitudes.
    In addition to self.training, we set another flag in this implement to control BN's behavior to train in eval mode.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(LP_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

    def forward(self, input, is_train):
        """
        We use is_train instead of self.training.
        """
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if is_train and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if is_train:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        return F.batch_norm(input, self.running_mean if not is_train or self.track_running_stats else None, self.running_var if not is_train or self.track_running_stats else None, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001, lin_probe=False, linear_type='standard', args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_mean_pooling = use_mean_pooling
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        elif args.sin_pos_emb:
            self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.lin_probe = lin_probe
        self.linear_type = linear_type
        if lin_probe:
            if self.linear_type == 'standard':
                self.fc_norm = None
            elif self.linear_type == 'attentive':
                self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.attentive_blocks = nn.ModuleList([AttentiveBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, init_values=0) for i in range(1)])
                self.fc_norm = LP_BatchNorm(embed_dim, affine=False)
        elif use_mean_pooling:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.fc_norm = None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.pos_embed is not None and use_abs_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.0):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).clone().detach()
            else:
                x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).clone().detach()
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        if self.lin_probe:
            if self.linear_type == 'standard':
                return x[:, 0]
            else:
                query_tokens = self.query_token.expand(batch_size, -1, -1)
                for blk in self.attentive_blocks:
                    query_tokens = blk(query_tokens, x, 0, 0, bool_masked_pos=None, rel_pos_bias=None)
                return self.fc_norm(query_tokens[:, 0, :], is_train=is_train)
        elif self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(t.mean(1))
        else:
            return x[:, 0]

    def forward(self, x, is_train=True):
        x = self.forward_features(x, is_train)
        x = self.head(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])
            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))
            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)
    return logger


DEFAULT_CACHE_DIR = '~/.cache'


ENV_MMCV_HOME = 'MMCV_HOME'


ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(os.getenv(ENV_MMCV_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))
    mkdir_or_exist(mmcv_home)
    return mmcv_home


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)
    return new_checkpoint


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0], 'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)
    return deprecate_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)
    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)
    return mmcls_urls


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def load_url_dist(url, model_dir=None, map_location='cpu'):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    return checkpoint


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        checkpoint = load_fileclient_dist(filename, backend='ceph', map_location=map_location)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_state_dict(model, state_dict, prefix='', ignore_missing='relative_position_index'):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(model, prefix=prefix)
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)
    missing_keys = warn_missing_keys
    if len(missing_keys) > 0:
        None
    if len(unexpected_keys) > 0:
        None
    if len(ignore_missing_keys) > 0:
        None
    if len(error_msgs) > 0:
        None


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning('Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    rank, _ = get_dist_info()
    if 'rel_pos_bias.relative_position_bias_table' in state_dict:
        if rank == 0:
            None
            num_layers = model.get_num_layers()
            rel_pos_bias = state_dict['rel_pos_bias.relative_position_bias_table']
            for i in range(num_layers):
                state_dict['blocks.%d.attn.relative_position_bias_table' % i] = rel_pos_bias.clone()
        state_dict.pop('rel_pos_bias.relative_position_bias_table')
    all_keys = list(state_dict.keys())
    all_keys = sorted(all_keys)
    None
    if all_keys[-2].startswith('encoder_to_decoder'):
        all_keys = [key for key in all_keys if key.startswith('encoder.')]
        None
        for key in all_keys:
            new_key = key.replace('encoder.', '')
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)
        for key in list(state_dict.keys()):
            if key.startswith('decoder.'):
                state_dict.pop(key)
        for key in list(state_dict.keys()):
            if key.startswith('norm.'):
                new_key = key.replace('norm.', 'fc_norm.')
                state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
    None
    for key in all_keys:
        if 'relative_position_index' in key:
            state_dict.pop(key)
        if 'relative_position_bias_table' in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if rank == 0:
                    None
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)
                left, right = 1.01, 1.5
                while right - left > 1e-06:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [(-_) for _ in reversed(dis)]
                x = r_ids + [0] + dis
                y = r_ids + [0] + dis
                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                if rank == 0:
                    None
                    None
                all_rel_pos_bias = []
                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(torch.Tensor(f(dx, dy)).contiguous().view(-1, 1))
                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            if rank == 0:
                None
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f'Error in loading {table_key}, pass')
        elif L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class BEiT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, use_abs_pos_emb=True, use_rel_pos_bias=False, use_sincos_pos_embed=True, use_shared_rel_pos_bias=False, out_indices=[3, 5, 7, 11], out_with_norm=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        elif use_sincos_pos_embed:
            self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.out_indices = out_indices
        if patch_size == 16:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), nn.SyncBatchNorm(embed_dim), nn.GELU(), nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        if not out_with_norm:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.0, decode=False):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            """
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).clone().detach()
            else:
                x = x[:,1:] + self.pos_embed.expand(batch_size, -1, -1).type_as(x[:,1:]).to(x.device).clone().detach()
                x = torch.cat([x[:,:1],x],dim=1)
            """
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            if i in self.out_indices:
                xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class FeatureSelectionModule(nn.Module):

    def __init__(self, in_c, out_c, norm='GM'):
        super(FeatureSelectionModule, self).__init__()
        self.conv_attn = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        xavier_init(self.conv_attn)
        for m in self.conv.modeuls():
            if isintance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        def forward(self, x):
            attn = self.sigmoid(self.conv_attn(F.avg_pool2d(x, x.size()[2:])))
            feat = torch.mul(x, attn)
            x = x + feat
            feat = self.conv(x)
            return feat


class FeatureAlign(nn.Module):

    def __init__(self, in_c, out_c, norm=None):
        super(FeatureAlign, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_c, out_c, norm='')
        self.relu = nn.ReLU(inplace=True)
        self.offset = nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.deform_conv2d = DeformConv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=8)

    def forward(self, teat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up], dim=1))
        feat_align = self.relu(self.deform_conv2d(feat_up, offset))
        return feat_align + feat_arm


class BEiT_FaPN(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, out_indices=[3, 5, 7, 11]):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.out_conv = []
        self.FaPN = []
        for i in range(len(out_indices)):
            self.out_conv.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False))
            self.FaPN.append(FeatureAlign(embed_dim, embed_dim))
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.out_indices = out_indices
        if patch_size == 16:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), nn.SyncBatchNorm(embed_dim), nn.GELU(), nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        features_fapn = []
        features_fapn.append(features[-1])
        for i in range(len(ops) - 1, 0, -1):
            new_feature = self.FaPN[i](features[i - 1], features[i])
            new_feature = self.out_conv[i](new_feature)
            features_fapn.append(new_feature)
        features_fapn.reverse()
        return tuple(features_fapn)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class CAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, use_abs_pos_emb=True, use_rel_pos_bias=False, use_sincos_pos_embed=True, use_shared_rel_pos_bias=False, out_indices=[3, 5, 7, 11], out_with_norm=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        elif use_sincos_pos_embed:
            self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.out_indices = out_indices
        if patch_size == 16:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), nn.SyncBatchNorm(embed_dim), nn.GELU(), nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        if not out_with_norm:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.0, decode=False):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            """
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).clone().detach()
            else:
                x = x[:,1:] + self.pos_embed.expand(batch_size, -1, -1).type_as(x[:,1:]).to(x.device).clone().detach()
                x = torch.cat([x[:,:1],x],dim=1)
            """
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            if i in self.out_indices:
                xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, out_indices=[3, 5, 7, 11]):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) for i in range(depth)])
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.out_indices = out_indices
        if patch_size == 16:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), nn.SyncBatchNorm(embed_dim), nn.GELU(), nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.0, decode=False):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            """
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).clone().detach()
            else:
                x = x[:,1:] + self.pos_embed.expand(batch_size, -1, -1).type_as(x[:,1:]).to(x.device).clone().detach()
                x = torch.cat([x[:,:1],x],dim=1)
            """
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class GlobalContextExtractor(nn.Module):
    """Global Context Extractor for CGNet.

    This class is employed to refine the joFint feature of both local feature
    and surrounding context.

    Args:
        channel (int): Number of input feature channels.
        reduction (int): Reductions for global context extractor. Default: 16.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, channel, reduction=16, with_cp=False):
        super(GlobalContextExtractor, self).__init__()
        self.channel = channel
        self.reduction = reduction
        assert reduction >= 1 and channel >= reduction
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):

        def _inner_forward(x):
            num_batch, num_channel = x.size()[:2]
            y = self.avg_pool(x).view(num_batch, num_channel)
            y = self.fc(y).view(num_batch, num_channel, 1, 1)
            return x * y
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
            Default: 2.
        reduction (int): Reduction for global context extractor. Default: 16.
        skip_connect (bool): Add input to output or not. Default: True.
        downsample (bool): Downsample the input to 1/2 or not. Default: False.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, skip_connect=True, downsample=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='PReLU'), with_cp=False):
        super(ContextGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample
        channels = out_channels if downsample else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        self.conv1x1 = ConvModule(in_channels, channels, kernel_size, stride, padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.f_loc = build_conv_layer(conv_cfg, channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.f_sur = build_conv_layer(conv_cfg, channels, channels, kernel_size=3, padding=dilation, groups=channels, dilation=dilation, bias=False)
        self.bn = build_norm_layer(norm_cfg, 2 * channels)[1]
        self.activate = nn.PReLU(2 * channels)
        if downsample:
            self.bottleneck = build_conv_layer(conv_cfg, 2 * channels, out_channels, kernel_size=1, bias=False)
        self.skip_connect = skip_connect and not downsample
        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)

    def forward(self, x):

        def _inner_forward(x):
            out = self.conv1x1(x)
            loc = self.f_loc(out)
            sur = self.f_sur(out)
            joi_feat = torch.cat([loc, sur], 1)
            joi_feat = self.bn(joi_feat)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                joi_feat = self.bottleneck(joi_feat)
            out = self.f_glo(joi_feat)
            if self.skip_connect:
                return x + out
            else:
                return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, num_downsampling):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class CGNet(nn.Module):
    """CGNet backbone.

    A Light-weight Context Guided Network for Semantic Segmentation
    arXiv: https://arxiv.org/abs/1811.08201

    Args:
        in_channels (int): Number of input image channels. Normally 3.
        num_channels (tuple[int]): Numbers of feature channels at each stages.
            Default: (32, 64, 128).
        num_blocks (tuple[int]): Numbers of CG blocks at stage 1 and stage 2.
            Default: (3, 21).
        dilations (tuple[int]): Dilation rate for surrounding context
            extractors at stage 1 and stage 2. Default: (2, 4).
        reductions (tuple[int]): Reductions for global context extractors at
            stage 1 and stage 2. Default: (8, 16).
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, in_channels=3, num_channels=(32, 64, 128), num_blocks=(3, 21), dilations=(2, 4), reductions=(8, 16), conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='PReLU'), norm_eval=False, with_cp=False):
        super(CGNet, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        assert isinstance(self.num_channels, tuple) and len(self.num_channels) == 3
        self.num_blocks = num_blocks
        assert isinstance(self.num_blocks, tuple) and len(self.num_blocks) == 2
        self.dilations = dilations
        assert isinstance(self.dilations, tuple) and len(self.dilations) == 2
        self.reductions = reductions
        assert isinstance(self.reductions, tuple) and len(self.reductions) == 2
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if 'type' in self.act_cfg and self.act_cfg['type'] == 'PReLU':
            self.act_cfg['num_parameters'] = num_channels[0]
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(ConvModule(cur_channels, num_channels[0], 3, 2 if i == 0 else 1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            cur_channels = num_channels[0]
        self.inject_2x = InputInjection(1)
        self.inject_4x = InputInjection(2)
        cur_channels += in_channels
        self.norm_prelu_0 = nn.Sequential(build_norm_layer(norm_cfg, cur_channels)[1], nn.PReLU(cur_channels))
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.level1.append(ContextGuidedBlock(cur_channels if i == 0 else num_channels[1], num_channels[1], dilations[0], reductions[0], downsample=i == 0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp))
        cur_channels = 2 * num_channels[1] + in_channels
        self.norm_prelu_1 = nn.Sequential(build_norm_layer(norm_cfg, cur_channels)[1], nn.PReLU(cur_channels))
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level2.append(ContextGuidedBlock(cur_channels if i == 0 else num_channels[2], num_channels[2], dilations[1], reductions[1], downsample=i == 0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, with_cp=with_cp))
        cur_channels = 2 * num_channels[2]
        self.norm_prelu_2 = nn.Sequential(build_norm_layer(norm_cfg, cur_channels)[1], nn.PReLU(cur_channels))

    def forward(self, x):
        output = []
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down2 = x
        x = self.norm_prelu_2(torch.cat([down2, x], 1))
        output.append(x)
        return output

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
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.PReLU):
                    constant_init(m, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Convert the model into training mode whill keeping the normalization
        layer freezed."""
        super(CGNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class LearningToDownsample(nn.Module):
    """Learning to downsample module.

    Args:
        in_channels (int): Number of input channels.
        dw_channels (tuple[int]): Number of output channels of the first and
            the second depthwise conv (dwconv) layers.
        out_channels (int): Number of output channels of the whole
            'learning to downsample' module.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
    """

    def __init__(self, in_channels, dw_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(LearningToDownsample, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        dw_channels1 = dw_channels[0]
        dw_channels2 = dw_channels[1]
        self.conv = ConvModule(in_channels, dw_channels1, 3, stride=2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.dsconv1 = DepthwiseSeparableConvModule(dw_channels1, dw_channels2, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg)
        self.dsconv2 = DepthwiseSeparableConvModule(dw_channels2, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        layers.extend([ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), ConvModule(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {input_h, input_w} is `x+1` and out size {output_h, output_w} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module.

    Args:
        in_channels (int): Number of input channels of the GFE module.
            Default: 64
        block_channels (tuple[int]): Tuple of ints. Each int specifies the
            number of output channels of each Inverted Residual module.
            Default: (64, 96, 128)
        out_channels(int): Number of output channels of the GFE module.
            Default: 128
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
            Default: 6
        num_blocks (tuple[int]): Tuple of ints. Each int specifies the
            number of times each Inverted Residual module is repeated.
            The repeated Inverted Residual modules are called a 'group'.
            Default: (3, 3, 3)
        strides (tuple[int]): Tuple of ints. Each int specifies
            the downsampling factor of each 'group'.
            Default: (2, 2, 1)
        pool_scales (tuple[int]): Tuple of ints. Each int specifies
            the parameter required in 'global average pooling' within PPM.
            Default: (1, 2, 3, 6)
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, expand_ratio=6, num_blocks=(3, 3, 3), strides=(2, 2, 1), pool_scales=(1, 2, 3, 6), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), align_corners=False):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert len(block_channels) == len(num_blocks) == 3
        self.bottleneck1 = self._make_layer(in_channels, block_channels[0], num_blocks[0], strides[0], expand_ratio)
        self.bottleneck2 = self._make_layer(block_channels[0], block_channels[1], num_blocks[1], strides[1], expand_ratio)
        self.bottleneck3 = self._make_layer(block_channels[1], block_channels[2], num_blocks[2], strides[2], expand_ratio)
        self.ppm = PPM(pool_scales, block_channels[2], block_channels[2] // 4, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=align_corners)
        self.out = ConvModule(block_channels[2] * 2, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, expand_ratio=6):
        layers = [InvertedResidual(in_channels, out_channels, stride, expand_ratio, norm_cfg=self.norm_cfg)]
        for i in range(1, blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1, expand_ratio, norm_cfg=self.norm_cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module.

    Args:
        higher_in_channels (int): Number of input channels of the
            higher-resolution branch.
        lower_in_channels (int): Number of input channels of the
            lower-resolution branch.
        out_channels (int): Number of output channels.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self, higher_in_channels, lower_in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), align_corners=False):
        super(FeatureFusionModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.dwconv = ConvModule(lower_in_channels, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv_lower_res = ConvModule(out_channels, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.conv_higher_res = ConvModule(higher_in_channels, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = resize(lower_res_feature, size=higher_res_feature.size()[2:], mode='bilinear', align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class FastSCNN(nn.Module):
    """Fast-SCNN Backbone.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        downsample_dw_channels (tuple[int]): Number of output channels after
            the first conv layer & the second conv layer in
            Learning-To-Downsample (LTD) module.
            Default: (32, 48).
        global_in_channels (int): Number of input channels of
            Global Feature Extractor(GFE).
            Equal to number of output channels of LTD.
            Default: 64.
        global_block_channels (tuple[int]): Tuple of integers that describe
            the output channels for each of the MobileNet-v2 bottleneck
            residual blocks in GFE.
            Default: (64, 96, 128).
        global_block_strides (tuple[int]): Tuple of integers
            that describe the strides (downsampling factors) for each of the
            MobileNet-v2 bottleneck residual blocks in GFE.
            Default: (2, 2, 1).
        global_out_channels (int): Number of output channels of GFE.
            Default: 128.
        higher_in_channels (int): Number of input channels of the higher
            resolution branch in FFM.
            Equal to global_in_channels.
            Default: 64.
        lower_in_channels (int): Number of input channels of  the lower
            resolution branch in FFM.
            Equal to global_out_channels.
            Default: 128.
        fusion_out_channels (int): Number of output channels of FFM.
            Default: 128.
        out_indices (tuple): Tuple of indices of list
            [higher_res_features, lower_res_features, fusion_output].
            Often set to (0,1,2) to enable aux. heads.
            Default: (0, 1, 2).
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self, in_channels=3, downsample_dw_channels=(32, 48), global_in_channels=64, global_block_channels=(64, 96, 128), global_block_strides=(2, 2, 1), global_out_channels=128, higher_in_channels=64, lower_in_channels=128, fusion_out_channels=128, out_indices=(0, 1, 2), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), align_corners=False):
        super(FastSCNN, self).__init__()
        if global_in_channels != higher_in_channels:
            raise AssertionError('Global Input Channels must be the same                                  with Higher Input Channels!')
        elif global_out_channels != lower_in_channels:
            raise AssertionError('Global Output Channels must be the same                                 with Lower Input Channels!')
        self.in_channels = in_channels
        self.downsample_dw_channels1 = downsample_dw_channels[0]
        self.downsample_dw_channels2 = downsample_dw_channels[1]
        self.global_in_channels = global_in_channels
        self.global_block_channels = global_block_channels
        self.global_block_strides = global_block_strides
        self.global_out_channels = global_out_channels
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.fusion_out_channels = fusion_out_channels
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.learning_to_downsample = LearningToDownsample(in_channels, downsample_dw_channels, global_in_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.global_feature_extractor = GlobalFeatureExtractor(global_in_channels, global_block_channels, global_out_channels, strides=self.global_block_strides, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=self.align_corners)
        self.feature_fusion = FeatureFusionModule(higher_in_channels, lower_in_channels, fusion_out_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=self.align_corners)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(higher_res_features, lower_res_features)
        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self, num_branches, blocks, num_blocks, in_channels, num_channels, multiscale_output=True, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True)):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        """Check branches configuration."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        """Build one branch."""
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(build_conv_layer(self.conv_cfg, self.in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(self.norm_cfg, num_channels[branch_index] * block.expansion)[1])
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index], stride, downsample=downsample, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index], with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Build multiple branch."""
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1], Upsample(scale_factor=2 ** (j - i), mode='bilinear', align_corners=False)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + resize(self.fuse_layers[i][j](x[j]), size=x[i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4], [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self, widen_factor=1.0, strides=(1, 2, 2, 2, 1, 2, 1), dilations=(1, 1, 1, 1, 1, 1, 1), out_indices=(1, 2, 4, 6), frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), norm_eval=False, with_cp=False):
        super(MobileNetV2, self).__init__()
        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError(f'the item in out_indices must in range(0, 8). But received {index}')
        if frozen_stages not in range(-1, 7):
            raise ValueError(f'frozen_stages must be in range(-1, 7). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(out_channels=out_channels, num_blocks=num_blocks, stride=stride, dilation=dilation, expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, dilation, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(InvertedResidual(self.in_channels, out_channels, stride if i == 0 else 1, expand_ratio=expand_ratio, dilation=dilation if i == 0 else 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class MobileNetV3(nn.Module):
    """MobileNetV3 backbone.

    This backbone is the improved implementation of `Searching for MobileNetV3
    <https://ieeexplore.ieee.org/document/9008835>`_.

    Args:
        arch (str): Architechture of mobilnetv3, from {'small', 'large'}.
            Default: 'small'.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (tuple[int]): Output from which layer.
            Default: (0, 1, 12).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defualt: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Defualt: False.
    """
    arch_settings = {'small': [[3, 16, 16, True, 'ReLU', 2], [3, 72, 24, False, 'ReLU', 2], [3, 88, 24, False, 'ReLU', 1], [5, 96, 40, True, 'HSwish', 2], [5, 240, 40, True, 'HSwish', 1], [5, 240, 40, True, 'HSwish', 1], [5, 120, 48, True, 'HSwish', 1], [5, 144, 48, True, 'HSwish', 1], [5, 288, 96, True, 'HSwish', 2], [5, 576, 96, True, 'HSwish', 1], [5, 576, 96, True, 'HSwish', 1]], 'large': [[3, 16, 16, False, 'ReLU', 1], [3, 64, 24, False, 'ReLU', 2], [3, 72, 24, False, 'ReLU', 1], [5, 72, 40, True, 'ReLU', 2], [5, 120, 40, True, 'ReLU', 1], [5, 120, 40, True, 'ReLU', 1], [3, 240, 80, False, 'HSwish', 2], [3, 200, 80, False, 'HSwish', 1], [3, 184, 80, False, 'HSwish', 1], [3, 184, 80, False, 'HSwish', 1], [3, 480, 112, True, 'HSwish', 1], [3, 672, 112, True, 'HSwish', 1], [5, 672, 160, True, 'HSwish', 2], [5, 960, 160, True, 'HSwish', 1], [5, 960, 160, True, 'HSwish', 1]]}

    def __init__(self, arch='small', conv_cfg=None, norm_cfg=dict(type='BN'), out_indices=(0, 1, 12), frozen_stages=-1, reduction_factor=1, norm_eval=False, with_cp=False):
        super(MobileNetV3, self).__init__()
        assert arch in self.arch_settings
        assert isinstance(reduction_factor, int) and reduction_factor > 0
        assert mmcv.is_tuple_of(out_indices, int)
        for index in out_indices:
            if index not in range(0, len(self.arch_settings[arch]) + 2):
                raise ValueError(f'the item in out_indices must in range(0, {len(self.arch_settings[arch]) + 2}). But received {index}')
        if frozen_stages not in range(-1, len(self.arch_settings[arch]) + 2):
            raise ValueError(f'frozen_stages must be in range(-1, {len(self.arch_settings[arch]) + 2}). But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.reduction_factor = reduction_factor
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []
        in_channels = 16
        layer = ConvModule(in_channels=3, out_channels=in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=dict(type='Conv2dAdaptivePadding'), norm_cfg=self.norm_cfg, act_cfg=dict(type='HSwish'))
        self.add_module('layer0', layer)
        layers.append('layer0')
        layer_setting = self.arch_settings[self.arch]
        for i, params in enumerate(layer_setting):
            kernel_size, mid_channels, out_channels, with_se, act, stride = params
            if self.arch == 'large' and i >= 12 or self.arch == 'small' and i >= 8:
                mid_channels = mid_channels // self.reduction_factor
                out_channels = out_channels // self.reduction_factor
            if with_se:
                se_cfg = dict(channels=mid_channels, ratio=4, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid', bias=3.0, divisor=6.0)))
            else:
                se_cfg = None
            layer = InvertedResidual(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size, stride=stride, se_cfg=se_cfg, with_expand_conv=in_channels != mid_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type=act), with_cp=self.with_cp)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        layer = ConvModule(in_channels=in_channels, out_channels=576 if self.arch == 'small' else 960, kernel_size=1, stride=1, dilation=4, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=dict(type='HSwish'))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)
        if self.arch == 'small':
            self.layer4.depthwise_conv.conv.stride = 1, 1
            self.layer9.depthwise_conv.conv.stride = 1, 1
            for i in range(4, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidual):
                    modified_module = layer.depthwise_conv.conv
                else:
                    modified_module = layer.conv
                if i < 9:
                    modified_module.dilation = 2, 2
                    pad = 2
                else:
                    modified_module.dilation = 4, 4
                    pad = 4
                if not isinstance(modified_module, Conv2dAdaptivePadding):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = pad, pad
        else:
            self.layer7.depthwise_conv.conv.stride = 1, 1
            self.layer13.depthwise_conv.conv.stride = 1, 1
            for i in range(7, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidual):
                    modified_module = layer.depthwise_conv.conv
                else:
                    modified_module = layer.conv
                if i < 13:
                    modified_module.dilation = 2, 2
                    pad = 2
                else:
                    modified_module.dilation = 4, 4
                    pad = 4
                if not isinstance(modified_module, Conv2dAdaptivePadding):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = pad, pad
        return layers

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv2d in ResNeSt.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels. Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        dcn (dict): Config dict for DCN. Default: None.
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        self.dcn = dcn
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if self.with_dcn and not fallback_on_stride:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            conv_cfg = dcn
        self.conv = build_conv_layer(conv_cfg, in_channels, channels * radix, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.norm0_name, norm0 = build_norm_layer(norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        """nn.Module: the normalization layer named "norm0" """
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.norm1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), multi_grid=None, contract_dilation=False, **kwargs):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=first_dilation, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation if multi_grid is None else multi_grid[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super(ResLayer, self).__init__(*layers)


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=stride if i == 0 else 1, dilation=1 if i == 0 else dilation, padding=1 if i == 0 else dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, kernel_size=4, scale_factor=2):
        super(DeconvModule, self).__init__()
        assert kernel_size - scale_factor >= 0 and (kernel_size - scale_factor) % 2 == 0, f'kernel_size should be greater than or equal to scale_factor and (kernel_size - scale_factor) should be even numbers, while the kernel size is {kernel_size} and scale_factor is {scale_factor}.'
        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsampe_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, conv_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, upsampe_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()
        self.with_cp = with_cp
        conv = ConvModule(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        upsample = nn.Upsample(**upsampe_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), dcn=None, plugins=None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(cfg=upsample_cfg, in_channels=in_channels, out_channels=skip_channels, with_cp=with_cp, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    """UNet backbone.
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    https://arxiv.org/pdf/1505.04597.pdf

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondance encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondance encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondance decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondance encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.

    Notice:
        The input image size should be devisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_devisible.

    """

    def __init__(self, in_channels=3, base_channels=64, num_stages=5, strides=(1, 1, 1, 1, 1), enc_num_convs=(2, 2, 2, 2, 2), dec_num_convs=(2, 2, 2, 2), downsamples=(True, True, True, True), enc_dilations=(1, 1, 1, 1, 1), dec_dilations=(1, 1, 1, 1), with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), norm_eval=False, dcn=None, plugins=None):
        super(UNet, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, f'The length of strides should be equal to num_stages, while the strides is {strides}, the length of strides is {len(strides)}, and the num_stages is {num_stages}.'
        assert len(enc_num_convs) == num_stages, f'The length of enc_num_convs should be equal to num_stages, while the enc_num_convs is {enc_num_convs}, the length of enc_num_convs is {len(enc_num_convs)}, and the num_stages is {num_stages}.'
        assert len(dec_num_convs) == num_stages - 1, f'The length of dec_num_convs should be equal to (num_stages-1), while the dec_num_convs is {dec_num_convs}, the length of dec_num_convs is {len(dec_num_convs)}, and the num_stages is {num_stages}.'
        assert len(downsamples) == num_stages - 1, f'The length of downsamples should be equal to (num_stages-1), while the downsamples is {downsamples}, the length of downsamples is {len(downsamples)}, and the num_stages is {num_stages}.'
        assert len(enc_dilations) == num_stages, f'The length of enc_dilations should be equal to num_stages, while the enc_dilations is {enc_dilations}, the length of enc_dilations is {len(enc_dilations)}, and the num_stages is {num_stages}.'
        assert len(dec_dilations) == num_stages - 1, f'The length of dec_dilations should be equal to (num_stages-1), while the dec_dilations is {dec_dilations}, the length of dec_dilations is {len(dec_dilations)}, and the num_stages is {num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = strides[i] != 1 or downsamples[i - 1]
                self.decoder.append(UpConvBlock(conv_block=BasicConvBlock, in_channels=base_channels * 2 ** i, skip_channels=base_channels * 2 ** (i - 1), out_channels=base_channels * 2 ** (i - 1), num_convs=dec_num_convs[i - 1], stride=1, dilation=dec_dilations[i - 1], with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_cfg=upsample_cfg if upsample else None, dcn=None, plugins=None))
            enc_conv_block.append(BasicConvBlock(in_channels=in_channels, out_channels=base_channels * 2 ** i, num_convs=enc_num_convs[i], stride=strides[i], dilation=enc_dilations[i], with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None))
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = base_channels * 2 ** i

    def forward(self, x):
        self._check_input_devisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_devisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert h % whole_downsample_rate == 0 and w % whole_downsample_rate == 0, f'The input image size {h, w} should be devisible by the whole downsample rate {whole_downsample_rate}, when num_stages is {self.num_stages}, strides is {self.strides}, and downsamples is {self.downsamples}.'

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
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


class PPMConcat(nn.ModuleList):
    """Pyramid Pooling Module that only concat the features of each layer.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
    """

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__([nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])

    def forward(self, feats):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = torch.cat(ppm_outs, dim=2)
        return concat_outs


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels, out_channels, share_key_query, query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm, value_out_norm, matmul_norm, with_out, conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(key_in_channels, channels, num_convs=key_query_num_convs, use_conv_module=key_query_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(query_in_channels, channels, num_convs=key_query_num_convs, use_conv_module=key_query_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.value_project = self.build_project(key_in_channels, channels if with_out else out_channels, num_convs=value_out_num_convs, use_conv_module=value_out_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(channels, out_channels, num_convs=value_out_num_convs, use_conv_module=value_out_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.out_project = None
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module, conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [ConvModule(in_channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)]
            for _ in range(num_convs - 1):
                convs.append(ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = self.channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class AFNB(nn.Module):
    """Asymmetric Fusion Non-local Block(AFNB)

    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
            and query projection.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels, out_channels, query_scales, key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(AFNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(SelfAttentionBlock(low_in_channels=low_in_channels, high_in_channels=high_in_channels, channels=channels, out_channels=out_channels, share_key_query=False, query_scale=query_scale, key_pool_scales=key_pool_scales, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = ConvModule(out_channels + high_in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, low_feats, high_feats):
        """Forward function."""
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, high_feats], 1))
        return output


class APNB(nn.Module):
    """Asymmetric Pyramid Non-local Block (APNB)

    Args:
        in_channels (int): Input channels of key/query feature,
            which is the key feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, in_channels, channels, out_channels, query_scales, key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(APNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(SelfAttentionBlock(low_in_channels=in_channels, high_in_channels=in_channels, channels=channels, out_channels=out_channels, share_key_query=True, query_scale=query_scale, key_pool_scales=key_pool_scales, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = ConvModule(2 * in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, feats):
        """Forward function."""
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output


class ACM(nn.Module):
    """Adaptive Context Module used in APCNet.

    Args:
        pool_scale (int): Pooling scale used in Adaptive Context
            Module to extract region fetures.
        fusion (bool): Add one conv to fuse residual feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ACM, self).__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.input_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.global_info = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.gla = nn.Conv2d(self.channels, self.pool_scale ** 2, 1, 1, 0)
        self.residual_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        if self.fusion:
            self.fusion_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        x = self.input_redu_conv(x)
        pooled_x = self.pooled_redu_conv(pooled_x)
        batch_size = x.size(0)
        pooled_x = pooled_x.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        affinity_matrix = self.gla(x + resize(self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:])).permute(0, 2, 3, 1).reshape(batch_size, -1, self.pool_scale ** 2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        z_out = torch.matmul(affinity_matrix, pooled_x)
        z_out = z_out.permute(0, 2, 1).contiguous()
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)
        return z_out


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(ConvModule(self.in_channels, self.channels, 1 if dilation == 1 else 3, dilation=dilation, padding=0 if dilation == 1 else dilation, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma(out) + x
        return out


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)


class DCM(nn.Module):
    """Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel
            used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, filter_size, fusion, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(DCM, self).__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.input_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        if self.norm_cfg is not None:
            self.norm = build_norm_layer(self.norm_cfg, self.channels)[1]
        else:
            self.norm = None
        self.activate = build_activation_layer(self.act_cfg)
        if self.fusion:
            self.fusion_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        generted_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        generted_filter = generted_filter.view(b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = pad, pad, pad, pad
        else:
            p2d = pad + 1, pad, pad + 1, pad
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x, weight=generted_filter, groups=b * c)
        output = output.view(b, c, h, w)
        if self.norm is not None:
            output = self.norm(output)
        output = self.activate(output)
        if self.fusion:
            output = self.fusion_conv(output)
        return output


def reduce_mean(tensor):
    """Reduce mean when distributed training."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
    """

    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2.0 / self.num_bases))
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)

    def forward(self, feats):
        """Forward function."""
        batch_size, channels, height, width = feats.size()
        feats = feats.view(batch_size, channels, height * width)
        bases = self.bases.repeat(batch_size, 1, 1)
        with torch.no_grad():
            for i in range(self.num_stages):
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                attention_normed = F.normalize(attention, dim=1, p=1)
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                bases = F.normalize(bases, dim=1, p=2)
        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)
        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = reduce_mean(bases)
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 - self.momentum) * self.bases + self.momentum * bases
        return feats_recon


class Encoding(nn.Module):
    """Encoding Layer: a learnable residual encoder.

    Input is of shape  (batch_size, channels, height, width).
    Output is of shape (batch_size, num_codes, channels).

    Args:
        channels: dimension of the features or feature channels
        num_codes: number of code words
    """

    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        self.channels, self.num_codes = channels, num_codes
        std = 1.0 / (num_codes * channels) ** 0.5
        self.codewords = nn.Parameter(torch.empty(num_codes, channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assigment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assigment_weights.unsqueeze(3) * (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        assigment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assigment_weights, x, self.codewords)
        return encoded_feat

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}x{self.channels})'
        return repr_str


class EncModule(nn.Module):
    """Encoding Module used in EncNet.

    Args:
        in_channels (int): Input channels.
        num_codes (int): Number of code words.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, in_channels, num_codes, conv_cfg, norm_cfg, act_cfg):
        super(EncModule, self).__init__()
        self.encoding_project = ConvModule(in_channels, in_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if norm_cfg is not None:
            encoding_norm_cfg = norm_cfg.copy()
            if encoding_norm_cfg['type'] in ['BN', 'IN']:
                encoding_norm_cfg['type'] += '1d'
            else:
                encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace('2d', '1d')
        else:
            encoding_norm_cfg = dict(type='BN1d')
        self.encoding = nn.Sequential(Encoding(channels=in_channels, num_codes=num_codes), build_norm_layer(encoding_norm_cfg, num_codes)[1], nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(self.in_channels, self.channels, 3, dilation=dilation, padding=dilation, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1,), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)
    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1
    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask
    return bin_labels, bin_label_weights


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert pred.dim() == 2 and label.dim() == 1 or pred.dim() == 4 and label.dim() == 3, 'Only pred shape [N, C], label shape [N] or pred shape [N, C, H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape, ignore_index)
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None, ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


def flatten_binary_logits(logits, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case) Remove labels equal to
    'ignore_index'."""
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = labels != ignore_index
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).

    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, classes='present', per_image=False, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [B, H, W], logits at each pixel
            (between -infty and +infty).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        classes (str | list[int], optional): Placeholder, to be consistent with
            other loss. Default: None.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): Placeholder, to be consistent
            with other loss. Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if per_image:
        loss = [lovasz_hinge_flat(*flatten_binary_logits(logit.unsqueeze(0), label.unsqueeze(0), ignore_index)) for logit, label in zip(logits, labels)]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_logits(logits, labels, ignore_index))
    return loss


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes choosed to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        return probs * 0.0
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


def lovasz_softmax(probs, labels, classes='present', per_image=False, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [B, C, H, W], class probabilities at each
            prediction (between 0 and 1).
        labels (torch.Tensor): [B, H, W], ground truth labels (between 0 and
            C - 1).
        classes (str | list[int], optional): Classes choosed to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if per_image:
        loss = [lovasz_softmax_flat(*flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index), classes=classes, class_weight=class_weight) for prob, label in zip(probs, labels)]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(*flatten_probs(probs, labels, ignore_index), classes=classes, class_weight=class_weight)
    return loss


class LovaszLoss(nn.Module):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        classes (str | list[int], optional): Classes choosed to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, loss_type='multi_class', classes='present', per_image=False, reduction='mean', class_weight=None, loss_weight=1.0):
        super(LovaszLoss, self).__init__()
        assert loss_type in ('binary', 'multi_class'), "loss_type should be                                                     'binary' or 'multi_class'."
        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ('all', 'present') or mmcv.is_list_of(classes, int)
        if not per_image:
            assert reduction == 'none', "reduction should be 'none' when                                                         per_image is False."
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, self.classes, self.per_image, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


class FPN(nn.Module):
    """Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=False, relu_before_extra_convs=False, no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, act_cfg=None, upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg if not self.no_norm_on_lateral else None, act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value
    return outputs


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self, channels, ratio=16, conv_cfg=None, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = act_cfg, act_cfg
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=channels, out_channels=make_divisible(channels // ratio, 8), kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=make_divisible(channels // ratio, 8), out_channels=channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class InvertedResidualV3(nn.Module):
    """Inverted Residual Block for MobileNetV3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernal size of the depthwise convolution.
            Default: 3.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Defaul: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, se_cfg=None, with_expand_conv=True, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), with_cp=False):
        super(InvertedResidualV3, self).__init__()
        self.with_res_shortcut = stride == 1 and in_channels == out_channels
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv
        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = ConvModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, conv_cfg=dict(type='Conv2dAdaptivePadding') if stride == 2 else conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            if self.with_expand_conv:
                out = self.expand_conv(out)
            out = self.depthwise_conv(out)
            if self.with_se:
                out = self.se(out)
            out = self.linear_conv(out)
            if self.with_res_shortcut:
                return x + out
            else:
                return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class VisionTransformerEncoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, init_values=None, attn_head_dim=None, use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim, use_cls_token=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=None, attn_head_dim=attn_head_dim) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.0, use_cls_token=False):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature ** omega
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        if not use_cls_token:
            pos_embed = nn.Parameter(pos_emb)
        else:
            pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
            pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, dim = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_unmasked = x[~bool_masked_pos].reshape(batch_size, -1, dim)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1, dim)
            pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape(batch_size, -1, dim)
            pos_embed_unmasked = torch.cat((pos_embed[:, :1], pos_embed_unmasked), dim=1)
            x_unmasked = x_unmasked + pos_embed_unmasked
        x_unmasked = self.pos_drop(x_unmasked)
        for blk in self.blocks:
            x_unmasked = blk(x_unmasked, bool_masked_pos)
        x_unmasked = self.norm(x_unmasked)
        return x_unmasked

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        return x


class RegressorBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones(dim), requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q), bool_masked_pos, k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))
        return x


class VisionTransformerNeck(nn.Module):

    def __init__(self, patch_size=16, num_classes=8192, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, init_values=None, num_patches=196, init_std=0.02, args=None, patch_shape=(14, 14)):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.args = args
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.regressor_blocks = nn.ModuleList([RegressorBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, args.decoder_depth)]
        self.decoder_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(args.decoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_std = init_std
        trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.regressor_blocks):
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight.data, layer_id + 1)
        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos):
        for blk in self.regressor_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1), bool_masked_pos)
        x_masked = self.norm(x_masked)
        latent_pred = x_masked
        x_masked = x_masked + pos_embed_masked
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm2(x_masked)
        logits = self.head(x_masked)
        return logits, latent_pred


class VisionTransformerForMaskedImageModeling(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, init_values=None, attn_head_dim=None, use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()
        self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
        self.teacher = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.patch_embed.num_patches
        self.pretext_neck = VisionTransformerNeck(patch_size=patch_size, num_classes=args.decoder_num_classes, embed_dim=args.decoder_embed_dim, depth=args.regressor_depth, num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, num_patches=self.num_patches, init_std=init_std, args=args)
        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)
        if not args.fix_init_weight:
            self.apply(self._init_weights)
        self._init_teacher()

    def _init_teacher(self):
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1.0 - base_momentum)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    """
    Input shape:
        x: [bs, 3, 224, 224]
        bool_masked_pos: [bs, num_patch * num_patch]
    """

    def forward(self, x, bool_masked_pos, return_all_tokens=None):
        batch_size = x.size(0)
        """
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        """
        x_unmasked = self.encoder(x, bool_masked_pos=bool_masked_pos)
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)
        """
        Alignment constraint
        """
        with torch.no_grad():
            latent_target = self.teacher(x, bool_masked_pos=~bool_masked_pos)
            latent_target = latent_target[:, 1:, :]
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))
            self.momentum_update(self.args.base_momentum)
        """
        Latent contextual regressor and decoder
        """
        b, num_visible_plus1, dim = x_unmasked.shape
        x_unmasked = x_unmasked[:, 1:, :]
        num_masked_patches = self.num_patches - (num_visible_plus1 - 1)
        pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches + 1, dim)
        pos_embed_masked = pos_embed[:, 1:][bool_masked_pos].reshape(batch_size, -1, dim)
        pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape(batch_size, -1, dim)
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)
        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos)
        logits = logits.view(-1, logits.shape[2])
        return logits, latent_pred, latent_target


class BasicVAE(nn.Module):

    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class ResBlock(nn.Module):

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan, 3, padding=1), nn.ReLU(), nn.Conv2d(chan, chan, 3, padding=1), nn.ReLU(), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class DiscreteVAE(BasicVAE):

    def __init__(self, image_size=256, num_tokens=512, codebook_dim=512, num_layers=3, num_resnet_blocks=2, hidden_dim=64, channels=3, smooth_l1_loss=False, temperature=0.9, straight_through=False, kl_div_loss_weight=0.0):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        hdim = hidden_dim
        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))
        enc_chans = [channels, *enc_chans]
        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))
        enc_layers = []
        dec_layers = []
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))
        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))
        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // 8

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    @torch.no_grad()
    @eval_decorator
    def get_codebook_probs(self, images, temp):
        logits = self.forward(images, return_logits=True)
        return nn.Softmax(dim=1)(logits / temp)

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(self, img, return_loss=False, return_recons=False, return_logits=False, temp=None):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'
        logits = self.encoder(img)
        if return_logits:
            return logits
        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits.float(), tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight).type_as(logits)
        out = self.decoder(sampled)
        if not return_loss:
            return out
        recon_loss = self.loss_fn(img, out)
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        _C = logits.size(-1)
        avg_probs = F.softmax(logits.contiguous().view(-1, _C), dim=-1, dtype=torch.float32).mean(0)
        diversity_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-06), dim=-1).mean()
        if not return_recons:
            return recon_loss, diversity_loss
        return recon_loss, diversity_loss, out


class Dalle_VAE(BasicVAE):

    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir, device):
        self.encoder = load_model(os.path.join(model_dir, 'encoder.pkl'), device)
        self.decoder = load_model(os.path.join(model_dir, 'decoder.pkl'), device)

    def decode(self, img_seq):
        bsz = img_seq.size()[0]
        img_seq = img_seq.view(bsz, self.image_size // 8, self.image_size // 8)
        z = F.one_hot(img_seq, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob, no_process=False):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            bsz, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(bsz, self.image_size // 8, self.image_size // 8, self.encoder.vocab_size)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()


class VGGAN(BasicVAE):

    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, weight_path, device):
        self.vqgan = torch.load(weight_path, map_location=device)

    def get_codebook_indices(self, images):
        _, _, [_, _, indices] = self.vqgan.encode(images)
        return indices


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CSyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CustomSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InputInjection,
     lambda: ([], {'num_downsampling': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PPMConcat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchMerging,
     lambda: ([], {'input_resolution': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RelativePositionBias,
     lambda: ([], {'window_size': [4, 4], 'num_heads': 4}),
     lambda: ([], {}),
     True),
    (ResBlock,
     lambda: ([], {'chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lxtGH_CAE(_paritybench_base):
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

