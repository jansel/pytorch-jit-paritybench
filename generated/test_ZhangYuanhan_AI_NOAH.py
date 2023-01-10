import sys
_module = sys.modules[__name__]
del sys
get_vtab1k = _module
setup = _module
task_adaptation = _module
adapt_and_eval = _module
data = _module
base = _module
base_test = _module
caltech = _module
caltech_test = _module
cars = _module
cars_test = _module
cifar = _module
cifar_test = _module
clevr = _module
clevr_test = _module
cub = _module
cub_test = _module
data_testing_lib = _module
diabetic_retinopathy = _module
diabetic_retinopathy_test = _module
dmlab = _module
dmlab_test = _module
dsprites = _module
dsprites_test = _module
dtd = _module
dtd_test = _module
eurosat = _module
eurosat_test = _module
food101 = _module
food101_test = _module
inaturalist = _module
inaturalist_test = _module
kitti = _module
kitti_test = _module
oxford_flowers102 = _module
oxford_flowers102_test = _module
oxford_iiit_pet = _module
oxford_iiit_pet_test = _module
patch_camelyon = _module
patch_camelyon_test = _module
resisc45 = _module
resisc45_test = _module
smallnorb = _module
smallnorb_test = _module
sun397 = _module
sun397_test = _module
svhn = _module
svhn_test = _module
data_loader = _module
data_loader_test = _module
loop = _module
loop_test = _module
model = _module
model_test = _module
registry = _module
registry_test = _module
test_utils = _module
trainer = _module
trainer_test = _module
config = _module
datasets = _module
imagenet_withhold = _module
samplers = _module
subImageNet = _module
utils = _module
Linear_super = _module
module = _module
adapter_super = _module
embedding_super = _module
layernorm_super = _module
multihead_super = _module
multihead_super_prompt = _module
prompt_tuning_super = _module
qkv_super = _module
qkv_super_prompt = _module
supernet_transformer_prompt = _module
supernet_vision_transformer_timm = _module
utils = _module
supernet_engine_prompt = _module
supernet_train_prompt = _module

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


import scipy


import scipy.io as sio


from torchvision import datasets


from torchvision import transforms


from torchvision.datasets.folder import ImageFolder


from torchvision.datasets.folder import default_loader


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import torch.distributed as dist


import math


import time


from collections import defaultdict


from collections import deque


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch import nn


from torch.nn import Parameter


import logging


from functools import partial


from collections import OrderedDict


from copy import deepcopy


import warnings


from itertools import repeat


from typing import Iterable


from typing import Optional


import random


import torch.backends.cudnn as cudnn


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    return sample_bias


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat([sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim=0)
    return sample_weight


class LinearSuper(nn.Linear):

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()
        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0
        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdapterSuper(nn.Module):

    def __init__(self, embed_dims, reduction_dims, drop_rate_adapter=0):
        super(AdapterSuper, self).__init__()
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)
        None
        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-06)
        self.apply(_init_weights)

    def set_sample_config(self, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim, :]
            self.sampled_bias_0 = self.ln1.bias[:self.sample_embed_dim]
            self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
            self.sampled_bias_1 = self.ln2.bias

    def forward(self, x, identity=None):
        if self.identity:
            return x
        out = F.linear(x, self.sampled_weight_0, self.sampled_bias_0)
        out = self.activate(out)
        out = self.dropout(out)
        out = F.linear(out, self.sampled_weight_1, self.sampled_bias_1)
        if identity is None:
            identity = x
        return identity + out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchembedSuper(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchembedSuper, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.sampled_scale
        return x

    def calc_sampled_param_num(self):
        return self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
            total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops


class LayerNormSuper(torch.nn.LayerNorm):

    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)
        self.super_embed_dim = super_embed_dim
        self.sample_embed_dim = None
        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class RelativePosition2D_super(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table_v = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))
        trunc_normal_(self.embeddings_table_v, std=0.02)
        trunc_normal_(self.embeddings_table_h, std=0.02)
        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim):
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = self.embeddings_table_h[:, :sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:, :sample_head_dim]

    def calc_sampled_param_num(self):
        return self.sample_embeddings_table_h.numel() + self.sample_embeddings_table_v.numel()

    def forward(self, length_q, length_k):
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat_v = range_vec_k[None, :] // int(length_q ** 0.5) - range_vec_q[:, None] // int(length_q ** 0.5)
        distance_mat_h = range_vec_k[None, :] % int(length_q ** 0.5) - range_vec_q[:, None] % int(length_q ** 0.5)
        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.max_relative_position, self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.max_relative_position, self.max_relative_position)
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0), 'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0), 'constant', 0)
        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        embeddings = self.sample_embeddings_table_v[final_mat_v] + self.sample_embeddings_table_h[final_mat_h]
        return embeddings


class qkv_super(nn.Linear):

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False, LoRA_dim=1024):
        super().__init__(super_in_dim, super_out_dim, bias=bias)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        self.scale = scale
        self.profiling = False
        self.super_LoRA_dim = LoRA_dim
        self.LoRA_a = nn.Parameter(torch.zeros(super_in_dim, LoRA_dim))
        nn.init.kaiming_uniform_(self.LoRA_a, a=math.sqrt(5))
        self.LoRA_b = nn.Parameter(torch.zeros(LoRA_dim, super_out_dim))

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim, sample_out_dim, sample_LoRA_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_LoRA_dim = sample_LoRA_dim
        self._sample_parameters()

    def _sample_parameters(self):
        if self.sample_LoRA_dim != 0:
            self.weight_with_LoRA = self.weight + (self.LoRA_a[:, :self.sample_LoRA_dim] @ self.LoRA_b[:self.sample_LoRA_dim, :]).T
            self.samples['weight'] = sample_weight(self.weight_with_LoRA, self.sample_in_dim, self.sample_out_dim)
        else:
            self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()
        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0
        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


class AttentionSuper(nn.Module):

    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, normalization=False, relative_position=False, num_patches=None, max_relative_position=14, scale=False, change_qkv=False, LoRA_dim=1024):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = super_embed_dim
        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias, LoRA_dim=LoRA_dim)
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(super_embed_dim // num_heads, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(super_embed_dim // num_heads, max_relative_position)
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None
        self.proj = LinearSuper(super_embed_dim, super_embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None, sample_LoRA_dim=None):
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5
        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5
        self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3 * self.sample_qk_embed_dim, sample_LoRA_dim=sample_LoRA_dim)
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
            self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)

    def calc_sampled_param_num(self):
        return 0

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * sequence_length / 2.0
            total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0
        return total_flops

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.sample_scale
        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)).transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.sample_num_heads, -1)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B, self.sample_num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)
        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
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


def gelu(x: torch.Tensor) ->torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, dropout=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, pre_norm=True, scale=False, relative_position=False, change_qkv=False, max_relative_position=14, visual_prompt_dim=1024, LoRA_dim=1024, adapter_dim=1024):
        super().__init__()
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.is_identity_layer = None
        self.attn = AttentionSuper(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv, max_relative_position=max_relative_position, LoRA_dim=LoRA_dim)
        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.activation_fn = gelu
        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)
        self.visual_prompt_token = nn.Parameter(torch.zeros(1, visual_prompt_dim, dim))
        trunc_normal_(self.visual_prompt_token, std=0.02)
        self.adapter = AdapterSuper(embed_dims=dim, reduction_dims=adapter_dim)

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None, sample_LoRA_dim=None, sample_adapter_dim=None, sample_last_prompt_tuning_dim=None, sample_prompt_tuning_dim=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer * 64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim, sample_LoRA_dim=sample_LoRA_dim)
        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)
        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.sample_visual_prompt_dim = sample_prompt_tuning_dim
        self.sample_last_prompt_tuning_dim = sample_last_prompt_tuning_dim
        self.sample_adapter_dim = sample_adapter_dim
        self.adapter.set_sample_config(sample_embed_dim=self.sample_adapter_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x
        B = x.shape[0]
        if self.sample_visual_prompt_dim == 0:
            residual = x
        else:
            visual_prompt_tokens = self.visual_prompt_token[:, :self.sample_visual_prompt_dim, :self.sample_embed_dim].expand(B, -1, -1)
            if self.sample_last_prompt_tuning_dim == 0:
                x = torch.cat((x, visual_prompt_tokens), dim=1)
            else:
                x = torch.cat((x[:, :-self.sample_last_prompt_tuning_dim, :], visual_prompt_tokens), dim=1)
            residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + self.adapter(x)
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class Vision_TransformerSuper(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, pre_norm=True, scale=False, gp=False, relative_position=False, change_qkv=False, abs_pos=True, max_relative_position=14, super_prompt_tuning_dim=1024, super_LoRA_dim=1024, super_adapter_dim=1024):
        super(Vision_TransformerSuper, self).__init__()
        self.super_embed_dim = embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.scale = scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.gp = gp
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], pre_norm=pre_norm, scale=self.scale, change_qkv=change_qkv, relative_position=relative_position, max_relative_position=max_relative_position, visual_prompt_dim=super_prompt_tuning_dim, LoRA_dim=super_LoRA_dim, adapter_dim=super_adapter_dim))
        self.super_prompt_tuning_dim = super_prompt_tuning_dim
        self.visual_prompt_token = nn.Parameter(torch.zeros(1, super_prompt_tuning_dim, embed_dim))
        trunc_normal_(self.visual_prompt_token, std=0.02)
        self.visual_prompt_token_pos_embed = nn.Parameter(torch.zeros(1, super_prompt_tuning_dim, embed_dim))
        trunc_normal_(self.visual_prompt_token_pos_embed, std=0.02)
        num_patches = self.patch_embed_super.num_patches
        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)
        self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_stages(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'prompt' not in name and 'LoRA' not in name:
                param.requires_grad = False
        total_para_nums = 0
        adapter_para_nums = 0
        LoRA_para_nums = 0
        vp_para_nums = 0
        for name, param in self.named_parameters():
            None
            if param.requires_grad:
                total_para_nums += param.numel()
                if 'adapter' in name:
                    adapter_para_nums += param.numel()
                elif 'LoRA' in name:
                    LoRA_para_nums += param.numel()
                elif 'prompt' in name:
                    vp_para_nums += param.numel()
        None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        self.sample_prompt_tuning_dim = config['visual_prompt_dim']
        self.sample_LoRA_dim = config['lora_dim']
        self.sample_adapter_dim = config['adapter_dim']
        for i, blocks in enumerate(self.blocks):
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False, sample_embed_dim=self.sample_embed_dim[i], sample_mlp_ratio=self.sample_mlp_ratio[i], sample_num_heads=self.sample_num_heads[i], sample_dropout=sample_dropout, sample_out_dim=self.sample_output_dim[i], sample_attn_dropout=sample_attn_dropout, sample_prompt_tuning_dim=self.sample_prompt_tuning_dim[i], sample_LoRA_dim=self.sample_LoRA_dim[i], sample_last_prompt_tuning_dim=self.sample_prompt_tuning_dim[i - 1] if i > 0 else self.sample_last_prompt_tuning_dim[0], sample_adapter_dim=self.sample_adapter_dim[i])
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())
        return sum(numels) + self.sample_embed_dim[0] * (2 + self.patch_embed_super.num_patches)

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.sample_prompt_tuning_dim[0] != 0:
            visual_prompt_tokens = self.visual_prompt_token[:, :self.sample_prompt_tuning_dim[0], :self.sample_embed_dim[0]].expand(B, -1, -1)
            visual_prompt_tokens = visual_prompt_tokens + self.visual_prompt_token_pos_embed[:, :self.sample_prompt_tuning_dim[0], :self.sample_embed_dim[0]]
            x = torch.cat((x, visual_prompt_tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.pre_norm:
            x = self.norm(x)
        if self.gp:
            return torch.mean(x[:, 1:], dim=1)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, LoRA_dim=1024, prefix_dim=1024, drop_rate_LoRA=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.super_LoRA_dim = LoRA_dim
        None
        if LoRA_dim > 0:
            self.LoRA_a = nn.Linear(dim, LoRA_dim, bias=False)
            nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            self.LoRA_b = nn.Linear(LoRA_dim, dim * 3, bias=False)
            nn.init.zeros_(self.LoRA_b.weight)
        None
        if prefix_dim > 0:
            self.prefix_tokens_key = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            self.prefix_tokens_value = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            nn.init.xavier_uniform_(self.prefix_tokens_key)
            nn.init.xavier_uniform_(self.prefix_tokens_value)
        self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
        drop_rate_prefix = drop_rate_LoRA
        self.prefix_drop = nn.Dropout(p=drop_rate_prefix)

    def set_sample_config(self, sample_LoRA_dim, sample_prefix_dim):
        self.sample_LoRA_dim = sample_LoRA_dim
        self.LoRA_identity = False
        if self.sample_LoRA_dim == 0:
            self.LoRA_identity = True
        else:
            self.LoRA_a_weight = self.LoRA_a.weight[:self.sample_LoRA_dim, :]
            self.LoRA_b_weight = self.LoRA_b.weight[:, :self.sample_LoRA_dim]
        self.sample_prefix_dim = sample_prefix_dim
        self.prefix_identity = False
        if self.sample_prefix_dim == 0:
            self.prefix_identity = True
        else:
            self.prefix_weight_key = self.prefix_tokens_key[:, :self.sample_prefix_dim, :]
            self.prefix_weight_value = self.prefix_tokens_value[:, :self.sample_prefix_dim, :]

    def calc_sampled_param_num(self):
        if self.sample_LoRA_dim == 0:
            return 0
        else:
            return self.LoRA_a_weight.numel() + self.LoRA_b_weight.numel()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.LoRA_identity == False:
            qkv_delta = F.linear(self.LoRA_drop(x), self.LoRA_a_weight)
            qkv_delta = F.linear(qkv_delta, self.LoRA_b_weight).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_delta, k_delta, v_delta = qkv_delta.unbind(0)
            q, k, v = q + q_delta, k + k_delta, v + v_delta
        if self.prefix_identity == False:
            prefix_weight_key = self.prefix_weight_key.expand(B, -1, -1)
            prefix_weight_value = self.prefix_weight_value.expand(B, -1, -1)
            k, v = torch.cat((k, prefix_weight_key), dim=1), torch.cat((v, prefix_weight_value), dim=1)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, visual_prompt_dim=1024, LoRA_dim=1024, adapter_dim=1024, prefix_dim=1024, drop_rate_LoRA=0, drop_rate_prompt=0, drop_rate_adapter=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, LoRA_dim=LoRA_dim, prefix_dim=prefix_dim, drop_rate_LoRA=drop_rate_LoRA)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.super_visual_prompt_dim = visual_prompt_dim
        None
        if visual_prompt_dim > 0:
            self.visual_prompt_token = nn.Parameter(torch.zeros(1, visual_prompt_dim, dim))
            nn.init.xavier_uniform_(self.visual_prompt_token)
        self.drop_prompt = nn.Dropout(p=drop_rate_prompt)
        self.adapter = AdapterSuper(embed_dims=dim, reduction_dims=adapter_dim, drop_rate_adapter=drop_rate_adapter)

    def set_sample_config(self, sample_LoRA_dim=None, sample_adapter_dim=None, sample_prefix_dim=None, sample_last_prompt_tuning_dim=None, sample_prompt_tuning_dim=None):
        self.attn.set_sample_config(sample_LoRA_dim=sample_LoRA_dim, sample_prefix_dim=sample_prefix_dim)
        self.sample_visual_prompt_dim = 0
        self.sample_last_prompt_tuning_dim = 0
        if self.super_visual_prompt_dim > 0:
            self.sample_visual_prompt_dim = sample_prompt_tuning_dim
            self.sample_last_prompt_tuning_dim = sample_last_prompt_tuning_dim
        self.sample_adapter_dim = sample_adapter_dim
        self.adapter.set_sample_config(sample_embed_dim=self.sample_adapter_dim)

    def calc_sampled_param_num(self):
        if self.sample_visual_prompt_dim != 0:
            sample_visual_prompt_param = self.visual_prompt_token[:, :self.sample_visual_prompt_dim, :].numel()
        else:
            sample_visual_prompt_param = 0
        return sample_visual_prompt_param

    def forward(self, x):
        B = x.shape[0]
        if self.sample_visual_prompt_dim != 0:
            visual_prompt_tokens = self.visual_prompt_token[:, :self.sample_visual_prompt_dim, :].expand(B, -1, -1)
            visual_prompt_tokens = self.drop_prompt(visual_prompt_tokens)
            if self.sample_last_prompt_tuning_dim == 0:
                x = torch.cat((x, visual_prompt_tokens), dim=1)
            else:
                x = torch.cat((x[:, :-self.sample_last_prompt_tuning_dim, :], visual_prompt_tokens), dim=1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.adapter(self.drop_path(self.mlp(self.norm2(x))))
        return x


def _init_vit_weights(module: nn.Module, name: str='', head_bias: float=0.0, jax_impl: bool=False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif jax_impl:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-06)
                else:
                    nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


_logger = logging.getLogger(__name__)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ZhangYuanhan_AI_NOAH(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

