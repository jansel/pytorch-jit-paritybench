import sys
_module = sys.modules[__name__]
del sys
backbone = _module
backbonev2 = _module
checkpoint = _module
cityscapes_detection = _module
cityscapes_instance = _module
coco_detection = _module
coco_instance = _module
coco_instance_semantic = _module
deepfashion = _module
lvis_v1_instance = _module
voc0712 = _module
wider_face = _module
default_runtime = _module
cascade_mask_rcnn_pvtv2_b2_fpn = _module
cascade_mask_rcnn_r50_fpn = _module
cascade_rcnn_r50_fpn = _module
fast_rcnn_r50_fpn = _module
faster_rcnn_r50_caffe_c4 = _module
faster_rcnn_r50_caffe_dc5 = _module
faster_rcnn_r50_fpn = _module
mask_rcnn_r50_caffe_c4 = _module
mask_rcnn_r50_fpn = _module
retinanet_r50_fpn = _module
rpn_r50_caffe_c4 = _module
rpn_r50_fpn = _module
ssd300 = _module
schedule_1x = _module
schedule_20e = _module
schedule_2x = _module
mask_rcnn_efficientformer_l1_fpn_1x_coco = _module
mask_rcnn_efficientformer_l3_fpn_1x_coco = _module
mask_rcnn_efficientformer_l7_fpn_1x_coco = _module
mask_rcnn_efficientformerv2_l_fpn_1x_coco = _module
mask_rcnn_efficientformerv2_s0_fpn_1x_coco = _module
mask_rcnn_efficientformerv2_s1_fpn_1x_coco = _module
mask_rcnn_efficientformerv2_s2_fpn_1x_coco = _module
checkpoint = _module
epoch_based_runner = _module
optimizer = _module
train = _module
test = _module
train = _module
main = _module
models = _module
efficientformer = _module
efficientformer_v2 = _module
run_with_submitit = _module
align_resize = _module
backbone = _module
backbonev2 = _module
ade20k = _module
fpn_r50 = _module
schedule_160k = _module
schedule_20k = _module
schedule_40k = _module
schedule_80k = _module
fpn_efficientformer_l1_ade20k_40k = _module
fpn_efficientformer_l3_ade20k_40k = _module
fpn_efficientformer_l7_ade20k_40k = _module
fpn_efficientformerv2_l_ade20k_40k = _module
fpn_efficientformerv2_s0_ade20k_40k = _module
fpn_efficientformerv2_s1_ade20k_40k = _module
fpn_efficientformerv2_s2_ade20k_40k = _module
analyze_logs = _module
benchmark = _module
browse_dataset = _module
chase_db1 = _module
cityscapes = _module
coco_stuff10k = _module
coco_stuff164k = _module
drive = _module
hrf = _module
pascal_context = _module
stare = _module
voc_aug = _module
deploy_test = _module
get_flops = _module
mit2mmseg = _module
swin2mmseg = _module
vit2mmseg = _module
onnx2tensorrt = _module
print_config = _module
publish_model = _module
pytorch2onnx = _module
pytorch2torchscript = _module
test = _module
mmseg2torchserve = _module
mmseg_handler = _module
test_torchserve = _module
train = _module
util = _module
datasets = _module
engine = _module
losses = _module
samplers = _module
utils = _module

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


import copy


import torch


import torch.nn as nn


import math


from typing import Dict


import itertools


import re


import time


import warnings


from collections import OrderedDict


import torchvision


from torch.optim import Optimizer


from torch.utils import model_zoo


import random


import numpy as np


import torch.distributed as dist


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


from typing import Any


from typing import Iterable


from typing import Optional


from typing import Union


import matplotlib.pyplot as plt


from functools import partial


import torch._C


import torch.serialization


from torch import nn


from torchvision import datasets


from torchvision import transforms


from torchvision.datasets.folder import ImageFolder


from torchvision.datasets.folder import default_loader


from torch.nn import functional as F


from collections import defaultdict


from collections import deque


class Attention(torch.nn.Module):

    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=16):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', torch.zeros(num_heads, 49))
        self.register_buffer('attention_bias_idxs', torch.ones(49, 49).long())
        self.attention_biases_seg = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = torch.nn.functional.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class LGQuery(torch.nn.Module):

    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim))
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(torch.nn.Module):

    def __init__(self, dim=384, key_dim=32, num_heads=16, attn_ratio=4, resolution=7, out_dim=None, act_layer=None):
        super().__init__()
        key_dim = 16
        num_heads = 8
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.resolution = resolution
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)
        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * self.key_dim))
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1), nn.BatchNorm2d(self.num_heads * self.d))
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d, kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d), nn.BatchNorm2d(self.num_heads * self.d))
        self.proj = nn.Sequential(act_layer(), nn.Conv2d(self.dh, self.out_dim, 1), nn.BatchNorm2d(self.out_dim))
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2), abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', torch.zeros(num_heads, 196))
        self.register_buffer('attention_bias_idxs', torch.ones(49, 196).long())
        self.attention_biases_seg = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W // 4).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        attn = q @ k * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = torch.nn.functional.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, H // 2, W // 2) + v_local
        out = self.proj(out)
        return out


class Embedding(nn.Module):

    def __init__(self, patch_size=3, stride=2, padding=1, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d, light=False, asub=False, resolution=None, act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub
        if self.light:
            self.new_proj = nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans), nn.BatchNorm2d(in_chans), nn.Hardswish(), nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(embed_dim))
            self.skip = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(embed_dim))
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim, resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Flat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class Meta3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-05):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-05):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""
    _schemes = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if prefix not in cls._schemes or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(f'{prefix} is already registered as a loader backend, add "force=True" if you want to override it')
        cls._schemes = OrderedDict(sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """
        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls
        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            loader (function): checkpoint loader
        """
        for p in cls._schemes:
            if path.startswith(p):
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger=None):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        mmcv.print_log(f'load checkpoint from {class_name[10:]} path: {filename}', logger)
        return checkpoint_loader(filename, map_location)


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Default: None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
           OrderedDict storing model weights or a dict containing other
           information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


expansion_ratios_L = {'0': [4, 4, 4, 4, 4], '1': [4, 4, 4, 4, 4], '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4]}


class Attention4D(torch.nn.Module):

    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim), nn.BatchNorm2d(dim))
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None
        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * self.key_dim))
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * self.key_dim))
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1), nn.BatchNorm2d(self.num_heads * self.d))
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d, kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d), nn.BatchNorm2d(self.num_heads * self.d))
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Sequential(act_layer(), nn.Conv2d(self.dh, dim, 1), nn.BatchNorm2d(dim))
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', torch.zeros(num_heads, 49))
        self.register_buffer('attention_bias_idxs', torch.ones(49, 49).long())
        self.attention_biases_seg = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H = H // 2
            W = W // 2
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        attn = q @ k * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = torch.nn.functional.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)
        x = attn @ v
        out = x.transpose(2, 3).reshape(B, self.dh, H, W) + v_local
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.proj(out)
        return out


class AttnFFN(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-05):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(AttnFFN(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, resolution=resolution, stride=stride))
        else:
            blocks.append(FFN(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)
    return blocks


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_chs // 2), act_layer(), nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_chs), act_layer())


class EfficientFormer(nn.Module):

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None, pool_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU, num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, fork_feat=False, init_cfg=None, pretrained=None, vit_num=0, distillation=True, resolution=512, e_ratios=expansion_ratios_L, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)
        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratios, act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, resolution=math.ceil(resolution / 2 ** (i + 2)), vit_num=vit_num, e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(Embedding(patch_size=down_patch_size, stride=down_stride, padding=down_pad, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1], resolution=math.ceil(resolution / 2 ** (i + 2)), asub=asub, act_layer=act_layer, norm_layer=norm_layer))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = _state_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                outs.append(x)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        return cls_out


def eformer_block(dim, index, layers, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(AttnFFN(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, resolution=resolution, stride=stride))
        else:
            blocks.append(FFN(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormerV2(nn.Module):

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None, pool_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU, num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, fork_feat=False, init_cfg=None, pretrained=None, vit_num=0, distillation=True, resolution=224, e_ratios=expansion_ratios_L, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)
        network = []
        for i in range(len(layers)):
            stage = eformer_block(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratios, act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, resolution=math.ceil(resolution / 2 ** (i + 2)), vit_num=vit_num, e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(Embedding(patch_size=down_patch_size, stride=down_stride, padding=down_pad, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1], resolution=math.ceil(resolution / 2 ** (i + 2)), asub=asub, act_layer=act_layer, norm_layer=norm_layer))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = _state_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        return cls_out


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        if outputs_kd is None:
            raise ValueError('When knowledge distillation is enabled, the model is expected to return a Tuple[Tensor, Tensor] with the output of the class_token and the dist_token')
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(F.log_softmax(outputs_kd / T, dim=1), F.log_softmax(teacher_outputs / T, dim=1), reduction='sum', log_target=True) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention4D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Flat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LGQuery,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'resolution1': 4, 'resolution2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Meta3D,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_snap_research_EfficientFormer(_paritybench_base):
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

