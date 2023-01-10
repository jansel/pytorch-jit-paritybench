import sys
_module = sys.modules[__name__]
del sys
data_trainer = _module
data_transform = _module
dataset = _module
mask_generator = _module
mixup = _module
model_pretrain = _module
model_trainer = _module
optimizer = _module
transformer = _module
utils = _module
video_transformer = _module
visualize_attention = _module
weight_init = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.utils.data.dataloader import DataLoader


from collections.abc import Sequence


import random


import math


import torch.nn.functional as F


from torchvision import transforms


from torchvision.transforms.functional import InterpolationMode


import time


import warnings


import torch.utils.data as data


import torch.optim as optim


import torchvision


from functools import partial


from torch import optim as optim


from torch.nn.modules.utils import _pair


import matplotlib.pyplot as plt


import torch.distributed as dist


import matplotlib


from matplotlib.patches import Polygon


from torch.utils.data import DataLoader


import re


class DropPath(nn.Module):

    def __init__(self, dropout_p=None):
        super(DropPath, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        return self.drop_path(x, self.dropout_p, self.training)

    def drop_path(self, x, dropout_p=0.0, training=False):
        if dropout_p == 0.0 or not training:
            return x
        keep_prob = 1 - dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape).type_as(x)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


@torch.no_grad()
def constant_init_(tensor, constant_value=0):
    nn.init.constant_(tensor, constant_value)


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


class ClassificationHead(nn.Module):
    """Classification head for Video Transformer.
	
	Args:
		num_classes (int): Number of classes to be classified.
		in_channels (int): Number of channels in input feature.
		init_std (float): Std value for Initiation. Defaults to 0.02.
		kwargs (dict, optional): Any keyword argument to be used to initialize
			the head.
	"""

    def __init__(self, num_classes, in_channels, init_std=0.02, eval_metrics='finetune', **kwargs):
        super().__init__()
        self.init_std = init_std
        self.eval_metrics = eval_metrics
        self.cls_head = nn.Linear(in_channels, num_classes)
        self.init_weights(self.cls_head)

    def init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            if self.eval_metrics == 'finetune':
                trunc_normal_(module.weight, std=self.init_std)
            else:
                module.weight.data.normal_(mean=0.0, std=0.01)
        if hasattr(module, 'bias') and module.bias is not None:
            constant_init_(module.bias, constant_value=0)

    def forward(self, x):
        cls_score = self.cls_head(x)
        return cls_score


@torch.no_grad()
def kaiming_init_(tensor, a=0, mode='fan_out', nonlinearity='relu', distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


class PatchEmbed(nn.Module):
    """Images to Patch Embedding.

	Args:
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		tube_size (int): Size of temporal field of one 3D patch.
		in_channels (int): Channel num of input features. Defaults to 3.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
	"""

    def __init__(self, img_size, patch_size, tube_size=2, in_channels=3, embed_dims=768, conv_type='Conv2d'):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        num_patches = self.img_size[1] // self.patch_size[1] * (self.img_size[0] // self.patch_size[0])
        assert (num_patches * self.patch_size[0] * self.patch_size[1] == self.img_size[0] * self.img_size[1], 'The image size H*W must be divisible by patch size')
        self.num_patches = num_patches
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)
        elif conv_type == 'Conv3d':
            self.projection = nn.Conv3d(in_channels, embed_dims, kernel_size=(tube_size, patch_size, patch_size), stride=(tube_size, patch_size, patch_size))
        else:
            raise TypeError(f'Unsupported conv layer type {conv_type}')
        self.init_weights(self.projection)

    def init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(module, 'bias') and module.bias is not None:
            constant_init_(module.bias, constant_value=0)

    def forward(self, x):
        layer_type = type(self.projection)
        if layer_type == nn.Conv3d:
            x = rearrange(x, 'b t c h w -> b c t h w')
            x = self.projection(x)
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        elif layer_type == nn.Conv2d:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.projection(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            raise TypeError(f'Unsupported conv layer type {layer_type}')
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class DividedTemporalAttentionWithPreNorm(nn.Module):
    """Temporal Attention in Divided Space Time Attention. 
		A warp for torch.nn.MultiheadAttention.

	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

    def __init__(self, embed_dims, num_heads, num_frames, use_cls_token, attn_drop=0.0, proj_drop=0.0, layer_drop=dict(type=DropPath, dropout_p=0.1), norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.use_cls_token = use_cls_token
        self.norm = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        dropout_p = layer_drop.pop('dropout_p')
        layer_drop = layer_drop.pop('type')
        self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()
        if not use_cls_token:
            self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.init_weights(self.temporal_fc)

    def init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            constant_init_(module.weight, constant_value=0)
        if hasattr(module, 'bias') and module.bias is not None:
            constant_init_(module.bias, constant_value=0)

    def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
        assert residual is None, 'Always adding the shortcut in the forward function'
        cls_token = query[:, 0, :].unsqueeze(1)
        if self.use_cls_token:
            residual = query
            query = query[:, 1:, :]
        else:
            query = query[:, 1:, :]
            residual = query
        b, n, d = query.size()
        p, t = n // self.num_frames, self.num_frames
        query = rearrange(query, 'b (p t) d -> (b p) t d', p=p, t=t)
        if self.use_cls_token:
            cls_token = repeat(cls_token, 'b n d -> b (p n) d', p=p)
            cls_token = rearrange(cls_token, 'b p d -> (b p) 1 d')
            query = torch.cat((cls_token, query), 1)
        query = self.norm(query)
        attn_out, attn_weights = self.attn(query)
        if return_attention:
            return attn_weights
        attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
        if not self.use_cls_token:
            attn_out = self.temporal_fc(attn_out)
        if self.use_cls_token:
            cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
            cls_token = rearrange(cls_token, '(b p) d -> b p d', b=b)
            cls_token = reduce(cls_token, 'b p d -> b 1 d', 'mean')
            attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
            attn_out = torch.cat((cls_token, attn_out), 1)
            new_query = residual + attn_out
        else:
            attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
            new_query = residual + attn_out
            new_query = torch.cat((cls_token, new_query), 1)
        return new_query


class DividedSpatialAttentionWithPreNorm(nn.Module):
    """Spatial Attention in Divided Space Time Attention.
		A warp for torch.nn.MultiheadAttention.
		
	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

    def __init__(self, embed_dims, num_heads, num_frames, use_cls_token, attn_drop=0.0, proj_drop=0.0, layer_drop=dict(type=DropPath, dropout_p=0.1), norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.use_cls_token = use_cls_token
        self.norm = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        dropout_p = layer_drop.pop('dropout_p')
        layer_drop = layer_drop.pop('type')
        self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
        assert residual is None, 'Always adding the shortcut in the forward function'
        cls_token = query[:, 0, :].unsqueeze(1)
        if self.use_cls_token:
            residual = query
            query = query[:, 1:, :]
        else:
            query = query[:, 1:, :]
            residual = query
        b, n, d = query.size()
        p, t = n // self.num_frames, self.num_frames
        query = rearrange(query, 'b (p t) d -> (b t) p d', p=p, t=t)
        if self.use_cls_token:
            cls_token = repeat(cls_token, 'b n d -> b (t n) d', t=t)
            cls_token = rearrange(cls_token, 'b t d -> (b t) 1 d')
            query = torch.cat((cls_token, query), 1)
        query = self.norm(query)
        attn_out, attn_weights = self.attn(query)
        if return_attention:
            return attn_weights
        attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
        if self.use_cls_token:
            cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
            cls_token = rearrange(cls_token, '(b t) d -> b t d', b=b)
            cls_token = reduce(cls_token, 'b t d -> b 1 d', 'mean')
            attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
            attn_out = torch.cat((cls_token, attn_out), 1)
            new_query = residual + attn_out
        else:
            attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
            new_query = residual + attn_out
            new_query = torch.cat((cls_token, new_query), 1)
        return new_query


class MultiheadAttentionWithPreNorm(nn.Module):
    """Implements MultiheadAttention with residual connection.
	
	Args:
		embed_dims (int): The embedding dimension.
		num_heads (int): Parallel attention heads.
		attn_drop (float): A Dropout layer on attn_output_weights.
			Default: 0.0.
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Default: 0.0.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
		batch_first (bool): When it is True,  Key, Query and Value are shape of
			(batch, n, embed_dim), otherwise (n, batch, embed_dim).
			 Default to False.
	"""

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm, layer_drop=dict(type=DropPath, dropout_p=0.0), batch_first=False, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.norm = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        dropout_p = layer_drop.pop('dropout_p')
        layer_drop = layer_drop.pop('type')
        self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

    def forward(self, query, key=None, value=None, residual=None, attn_mask=None, key_padding_mask=None, return_attention=False, **kwargs):
        residual = query
        query = self.norm(query)
        attn_out, attn_weights = self.attn(query)
        if return_attention:
            return attn_weights
        new_query = residual + self.layer_drop(self.proj_drop(attn_out))
        return new_query


class FFNWithPreNorm(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.
	
	Args:
		embed_dims (int): The feature dimension. Same as
			`MultiheadAttention`. Defaults: 256.
		hidden_channels (int): The hidden dimension of FFNs.
			Defaults: 1024.
		num_layers (int, optional): The number of fully-connected layers in
			FFNs. Default: 2.
		act_layer (dict, optional): The activation layer for FFNs.
			Default: nn.GELU
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		dropout_p (float, optional): Probability of an element to be
			zeroed in FFN. Default 0.0.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
	"""

    def __init__(self, embed_dims=256, hidden_channels=1024, num_layers=2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, dropout_p=0.0, layer_drop=None, **kwargs):
        super().__init__()
        assert num_layers >= 2, f'num_layers should be no less than 2. got {num_layers}.'
        self.embed_dims = embed_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.norm = norm_layer(embed_dims)
        layers = []
        in_channels = embed_dims
        for _ in range(num_layers - 1):
            layers.append(nn.Sequential(nn.Linear(in_channels, hidden_channels), act_layer(), nn.Dropout(dropout_p)))
            in_channels = hidden_channels
        layers.append(nn.Linear(hidden_channels, embed_dims))
        layers.append(nn.Dropout(dropout_p))
        self.layers = nn.ModuleList(layers)
        if layer_drop:
            dropout_p = layer_drop.pop('dropout_p')
            layer_drop = layer_drop.pop('type')
            self.layer_drop = layer_drop(dropout_p)
        else:
            self.layer_drop = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return residual + self.layer_drop(x)


class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dims, num_heads, num_frames, hidden_channels, operator_order, norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_layers=2, dpr=0):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.ffns = nn.ModuleList([])
        for i, operator in enumerate(operator_order):
            if operator == 'self_attn':
                self.attentions.append(MultiheadAttentionWithPreNorm(embed_dims=embed_dims, num_heads=num_heads, batch_first=True, norm_layer=nn.LayerNorm, layer_drop=dict(type=DropPath, dropout_p=dpr)))
            elif operator == 'time_attn':
                self.attentions.append(DividedTemporalAttentionWithPreNorm(embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, use_cls_token=i == len(operator_order) - 2, layer_drop=dict(type=DropPath, dropout_p=dpr)))
            elif operator == 'space_attn':
                self.attentions.append(DividedSpatialAttentionWithPreNorm(embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, use_cls_token=i == len(operator_order) - 2, layer_drop=dict(type=DropPath, dropout_p=dpr)))
            elif operator == 'ffn':
                self.ffns.append(FFNWithPreNorm(embed_dims=embed_dims, hidden_channels=hidden_channels, num_layers=num_layers, act_layer=act_layer, norm_layer=norm_layer, layer_drop=dict(type=DropPath, dropout_p=dpr)))
            else:
                raise TypeError(f'Unsupported operator type {operator}')

    def forward(self, x, return_attention=False):
        attention_idx = 0
        for layer in self.attentions:
            if attention_idx >= len(self.attentions) - 1 and return_attention:
                x = layer(x, return_attention=True)
                return x
            else:
                x = layer(x)
            attention_idx += 1
        for layer in self.ffns:
            x = layer(x)
        return x


class TransformerContainer(nn.Module):

    def __init__(self, num_transformer_layers, embed_dims, num_heads, num_frames, hidden_channels, operator_order, drop_path_rate=0.1, norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_transformer_layers = num_transformer_layers
        dpr = np.linspace(0, drop_path_rate, num_transformer_layers)
        for i in range(num_transformer_layers):
            self.layers.append(BasicTransformerBlock(embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, hidden_channels=hidden_channels, operator_order=operator_order, norm_layer=norm_layer, act_layer=act_layer, num_layers=num_layers, dpr=dpr[i]))

    def forward(self, x, return_attention=False):
        layer_idx = 0
        for layer in self.layers:
            if layer_idx >= self.num_transformer_layers - 1 and return_attention:
                x = layer(x, return_attention=True)
            else:
                x = layer(x)
            layer_idx += 1
        return x


def get_sine_cosine_pos_emb(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def replace_state_dict(state_dict):
    for old_key in list(state_dict.keys()):
        if old_key.startswith('model'):
            new_key = old_key[6:]
            if 'in_proj' in new_key:
                new_key = new_key.replace('in_proj_', 'qkv.')
            elif 'out_proj' in new_key:
                new_key = new_key.replace('out_proj', 'proj')
            state_dict[new_key] = state_dict.pop(old_key)
        else:
            new_key = old_key[9:]
            state_dict[new_key] = state_dict.pop(old_key)


def init_from_kinetics_pretrain_(module, pretrain_pth):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrain_pth)
    else:
        state_dict = torch.load(pretrain_pth, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    replace_state_dict(state_dict)
    msg = module.load_state_dict(state_dict, strict=False)
    print_on_rank_zero(msg)


@torch.no_grad()
def init_from_vit_pretrain_(module, pretrained, conv_type, attention_type, copy_strategy, extend_strategy='temporal_avg', tube_size=2, num_time_transformer_layers=4):
    if isinstance(pretrained, str):
        if torch.cuda.is_available():
            state_dict = torch.load(pretrained)
        else:
            state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        old_state_dict_keys = list(state_dict.keys())
        for old_key in old_state_dict_keys:
            if conv_type == 'Conv3d':
                if 'patch_embed.projection.weight' in old_key:
                    weight = state_dict[old_key]
                    new_weight = repeat(weight, 'd c h w -> d c t h w', t=tube_size)
                    if extend_strategy == 'temporal_avg':
                        new_weight = new_weight / tube_size
                    elif extend_strategy == 'center_frame':
                        new_weight.zero_()
                        new_weight[:, :, tube_size // 2, :, :] = weight
                    state_dict[old_key] = new_weight
                    continue
            if attention_type == 'fact_encoder':
                new_key = old_key.replace('transformer_layers.layers', 'transformer_layers.0.layers')
            else:
                new_key = old_key
            if 'in_proj' in new_key:
                new_key = new_key.replace('in_proj_', 'qkv.')
            elif 'out_proj' in new_key:
                new_key = new_key.replace('out_proj', 'proj')
            if 'norms' in new_key:
                new_key = new_key.replace('norms.0', 'attentions.0.norm')
                new_key = new_key.replace('norms.1', 'ffns.0.norm')
            state_dict[new_key] = state_dict.pop(old_key)
        old_state_dict_keys = list(state_dict.keys())
        for old_key in old_state_dict_keys:
            if attention_type == 'divided_space_time':
                if 'attentions.0' in old_key:
                    new_key = old_key.replace('attentions.0', 'attentions.1')
                    if copy_strategy == 'repeat':
                        state_dict[new_key] = state_dict[old_key].clone()
                    elif copy_strategy == 'set_zero':
                        state_dict[new_key] = state_dict[old_key].clone().zero_()
            elif attention_type == 'fact_encoder':
                pattern = re.compile('(?<=layers.)\\d+')
                matchObj = pattern.findall(old_key)
                if len(matchObj) > 1 and int(matchObj[1]) < num_time_transformer_layers:
                    new_key = old_key.replace('transformer_layers.0.layers', 'transformer_layers.1.layers')
                    if copy_strategy == 'repeat':
                        state_dict[new_key] = state_dict[old_key].clone()
                    elif copy_strategy == 'set_zero':
                        state_dict[new_key] = state_dict[old_key].clone().zero_()
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
        print_on_rank_zero(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')


class TimeSformer(nn.Module):
    """TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
	Video Understanding? <https://arxiv.org/abs/2102.05095>`_

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to
			12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv2d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'space_only' and 'joint_space_time'.
			Defaults to 'divided_space_time'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
    supported_attention_types = ['divided_space_time', 'space_only', 'joint_space_time']

    def __init__(self, num_frames, img_size=224, patch_size=16, pretrain_pth=None, weights_from='imagenet', embed_dims=768, num_heads=12, num_transformer_layers=12, in_channels=3, conv_type='Conv2d', dropout_p=0.0, attention_type='divided_space_time', norm_layer=nn.LayerNorm, copy_strategy='repeat', use_learnable_pos_emb=True, return_cls_token=True, **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, f'Unsupported Attention Type {attention_type}!'
        self.num_frames = num_frames
        self.pretrain_pth = pretrain_pth
        self.weights_from = weights_from
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.copy_strategy = copy_strategy
        self.conv_type = conv_type
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.return_cls_token = return_cls_token
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims, conv_type=conv_type)
        num_patches = self.patch_embed.num_patches
        if self.attention_type == 'divided_space_time':
            operator_order = ['time_attn', 'space_attn', 'ffn']
            container = TransformerContainer(num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=operator_order)
            transformer_layers = container
        else:
            operator_order = ['self_attn', 'ffn']
            container = TransformerContainer(num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=operator_order)
            transformer_layers = container
        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-06)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
        if self.use_cls_token_temporal:
            num_frames = num_frames + 1
        else:
            num_patches = num_patches + 1
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        else:
            self.pos_embed = get_sine_cosine_pos_emb(num_patches, embed_dims)
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        if self.attention_type != 'space_only':
            if use_learnable_pos_emb:
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
            else:
                self.time_embed = get_sine_cosine_pos_emb(num_frames, embed_dims)
            self.drop_after_time = nn.Dropout(p=dropout_p)
        self.init_weights()

    def init_weights(self):
        if self.use_learnable_pos_emb:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.attention_type != 'space_only':
                nn.init.trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.pretrain_pth is not None:
            if self.weights_from == 'imagenet':
                init_from_vit_pretrain_(self, self.pretrain_pth, self.conv_type, self.attention_type, self.copy_strategy)
            elif self.weights_from == 'kinetics':
                init_from_kinetics_pretrain_(self, self.pretrain_pth)
            else:
                raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        b, t, c, h, w = x.shape
        x = self.patch_embed(x)
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        if self.use_cls_token_temporal:
            if self.use_learnable_pos_emb:
                x = x + self.pos_embed
            else:
                x = x + self.pos_embed.type_as(x).detach()
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            if self.use_learnable_pos_emb:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.interpolate_pos_encoding(x, w, h).type_as(x).detach()
        x = self.drop_after_pos(x)
        if self.attention_type != 'space_only':
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            if self.use_cls_token_temporal:
                x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
                cls_tokens = repeat(cls_tokens, 'b ... -> (repeat b) ...', repeat=x.shape[0] // b)
                x = torch.cat((cls_tokens, x), dim=1)
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.type_as(x).detach()
                cls_tokens = x[:b, 0, :].unsqueeze(1)
                x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.type_as(x).detach()
                x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            x = self.drop_after_time(x)
        return x, b

    def forward(self, x):
        x, b = self.prepare_tokens(x)
        x = self.transformer_layers(x)
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) p d -> b t p d', b=b)
            x = reduce(x, 'b t p d -> b p d', 'mean')
        x = self.norm(x)
        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)

    def get_last_selfattention(self, x):
        x, b = self.prepare_tokens(x)
        x = self.transformer_layers(x, return_attention=True)
        return x


class ViViT(nn.Module):
    """ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
    supported_attention_types = ['fact_encoder', 'joint_space_time', 'divided_space_time']

    def __init__(self, num_frames, img_size=224, patch_size=16, pretrain_pth=None, weights_from='imagenet', embed_dims=768, num_heads=12, num_transformer_layers=12, in_channels=3, dropout_p=0.0, tube_size=2, conv_type='Conv3d', attention_type='fact_encoder', norm_layer=nn.LayerNorm, copy_strategy='repeat', extend_strategy='temporal_avg', use_learnable_pos_emb=True, return_cls_token=True, **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, f'Unsupported Attention Type {attention_type}!'
        num_frames = num_frames // tube_size
        self.num_frames = num_frames
        self.pretrain_pth = pretrain_pth
        self.weights_from = weights_from
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.copy_strategy = copy_strategy
        self.extend_strategy = extend_strategy
        self.tube_size = tube_size
        self.num_time_transformer_layers = 0
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.return_cls_token = return_cls_token
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims, tube_size=tube_size, conv_type=conv_type)
        num_patches = self.patch_embed.num_patches
        if self.attention_type == 'divided_space_time':
            operator_order = ['time_attn', 'space_attn', 'ffn']
            container = TransformerContainer(num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=operator_order)
            transformer_layers = container
        elif self.attention_type == 'joint_space_time':
            operator_order = ['self_attn', 'ffn']
            container = TransformerContainer(num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=operator_order)
            transformer_layers = container
        else:
            transformer_layers = nn.ModuleList([])
            self.num_time_transformer_layers = 4
            spatial_transformer = TransformerContainer(num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=['self_attn', 'ffn'])
            temporal_transformer = TransformerContainer(num_transformer_layers=self.num_time_transformer_layers, embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, norm_layer=norm_layer, hidden_channels=embed_dims * 4, operator_order=['self_attn', 'ffn'])
            transformer_layers.append(spatial_transformer)
            transformer_layers.append(temporal_transformer)
        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-06)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        if attention_type == 'fact_encoder':
            num_frames = num_frames + 1
            num_patches = num_patches + 1
            self.use_cls_token_temporal = False
        else:
            self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
            if self.use_cls_token_temporal:
                num_frames = num_frames + 1
            else:
                num_patches = num_patches + 1
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
        else:
            self.pos_embed = get_sine_cosine_pos_emb(num_patches, embed_dims)
            self.time_embed = get_sine_cosine_pos_emb(num_frames, embed_dims)
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)
        self.init_weights()

    def init_weights(self):
        if self.use_learnable_pos_emb:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.pretrain_pth is not None:
            if self.weights_from == 'imagenet':
                init_from_vit_pretrain_(self, self.pretrain_pth, self.conv_type, self.attention_type, self.copy_strategy, self.extend_strategy, self.tube_size, self.num_time_transformer_layers)
            elif self.weights_from == 'kinetics':
                init_from_kinetics_pretrain_(self, self.pretrain_pth)
            else:
                raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def prepare_tokens(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        if self.use_cls_token_temporal:
            if self.use_learnable_pos_emb:
                x = x + self.pos_embed
            else:
                x = x + self.pos_embed.type_as(x).detach()
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            if self.use_learnable_pos_emb:
                x = x + self.pos_embed
            else:
                x = x + self.pos_embed.type_as(x).detach()
        x = self.drop_after_pos(x)
        if self.attention_type != 'fact_encoder':
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            if self.use_cls_token_temporal:
                x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
                cls_tokens = repeat(cls_tokens, 'b ... -> (repeat b) ...', repeat=x.shape[0] // b)
                x = torch.cat((cls_tokens, x), dim=1)
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.type_as(x).detach()
                cls_tokens = x[:b, 0, :].unsqueeze(1)
                x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.type_as(x).detach()
                x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            x = self.drop_after_time(x)
        return x, cls_tokens, b

    def forward(self, x):
        x, cls_tokens, b = self.prepare_tokens(x)
        if self.attention_type != 'fact_encoder':
            x = self.transformer_layers(x)
        else:
            spatial_transformer, temporal_transformer = *self.transformer_layers,
            x = spatial_transformer(x)
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
            x = reduce(x, 'b t p d -> b t d', 'mean')
            x = torch.cat((cls_tokens, x), dim=1)
            if self.use_learnable_pos_emb:
                x = x + self.time_embed
            else:
                x = x + self.time_embed.type_as(x).detach()
            x = self.drop_after_time(x)
            x = temporal_transformer(x)
        x = self.norm(x)
        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)

    def get_last_selfattention(self, x):
        x, cls_tokens, b = self.prepare_tokens(x)
        if self.attention_type != 'fact_encoder':
            x = self.transformer_layers(x, return_attention=True)
        else:
            spatial_transformer, temporal_transformer = *self.transformer_layers,
            x = spatial_transformer(x)
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
            x = reduce(x, 'b t p d -> b t d', 'mean')
            x = torch.cat((cls_tokens, x), dim=1)
            if self.use_learnable_pos_emb:
                x = x + self.time_embed
            else:
                x = x + self.time_embed.type_as(x).detach()
            x = self.drop_after_time(x)
            None
            x = temporal_transformer(x, return_attention=True)
        return x


class PatchEmbeding(nn.Module):
    """
	Transformer basic patch embedding module. Performs patchifying input, flatten and
	and transpose.
	The builder can be found in `create_patch_embed`.
	"""

    def __init__(self, *, patch_model=None):
        super().__init__()
        set_attributes(self, locals())
        assert self.patch_model is not None

    def forward(self, x):
        x = self.patch_model(x)
        return x.flatten(2).transpose(1, 2)


def create_conv_patch_embed(*, in_channels, out_channels, conv_kernel_size=(1, 16, 16), conv_stride=(1, 4, 4), conv_padding=(1, 7, 7), conv_bias=True, conv=nn.Conv3d):
    """
	Creates the transformer basic patch embedding. It performs Convolution, flatten and
	transpose.
	Args:
		in_channels (int): input channel size of the convolution.
		out_channels (int): output channel size of the convolution.
		conv_kernel_size (tuple): convolutional kernel size(s).
		conv_stride (tuple): convolutional stride size(s).
		conv_padding (tuple): convolutional padding size(s).
		conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
			output.
		conv (callable): Callable used to build the convolution layer.
	Returns:
		(nn.Module): transformer patch embedding layer.
	"""
    conv_module = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=conv_bias)
    return PatchEmbeding(patch_model=conv_module)


def create_multiscale_vision_transformers(*, spatial_size, temporal_size, cls_embed_on=True, sep_pos_embed=True, depth=16, norm='layernorm', input_channels=3, patch_embed_dim=96, conv_patch_embed_kernel=(3, 7, 7), conv_patch_embed_stride=(2, 4, 4), conv_patch_embed_padding=(1, 3, 3), enable_patch_embed_norm=False, use_2d_patch=False, num_heads=1, mlp_ratio=4.0, qkv_bias=True, dropout_rate_block=0.0, droppath_rate_block=0.0, pooling_mode='conv', pool_first=False, residual_pool=False, depthwise_conv=True, bias_on=True, separate_qkv=True, embed_dim_mul=None, atten_head_mul=None, pool_q_stride_size=None, pool_kv_stride_size=None, pool_kv_stride_adaptive=None, pool_kvq_kernel=None, head=None) ->nn.Module:
    """
	Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
	(ViT) is a specific case of MViT that only uses a single scale attention block.
	"""
    if use_2d_patch:
        assert temporal_size == 1, 'If use_2d_patch, temporal_size needs to be 1.'
    if pool_kv_stride_adaptive is not None:
        assert pool_kv_stride_size is None, 'pool_kv_stride_size should be none if pool_kv_stride_adaptive is set.'
    if norm == 'layernorm':
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        block_norm_layer = partial(nn.LayerNorm, eps=1e-06)
        attn_norm_layer = partial(nn.LayerNorm, eps=1e-06)
    else:
        raise NotImplementedError('Only supports layernorm.')
    if isinstance(spatial_size, int):
        spatial_size = spatial_size, spatial_size
    conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d
    norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None
    patch_embed = None
    input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
    input_stirde = (1,) + tuple(conv_patch_embed_stride) if use_2d_patch else conv_patch_embed_stride
    patch_embed_shape = [(input_dims[i] // input_stirde[i]) for i in range(len(input_dims))]
    cls_positional_encoding = SpatioTemporalClsPositionalEncoding(embed_dim=patch_embed_dim, patch_embed_shape=patch_embed_shape, sep_pos_embed=sep_pos_embed, has_cls=cls_embed_on)
    dpr = [x.item() for x in torch.linspace(0, droppath_rate_block, depth)]
    if dropout_rate_block > 0.0:
        pos_drop = nn.Dropout(p=dropout_rate_block)
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]
    mvit_blocks = nn.ModuleList()
    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]
    if pool_q_stride_size is not None:
        for i in range(len(pool_q_stride_size)):
            stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride_size[i][0]] = [(s + 1 if s > 1 else s) for s in pool_q_stride_size[i][1:]]
    if pool_kv_stride_adaptive is not None:
        _stride_kv = pool_kv_stride_adaptive
        pool_kv_stride_size = []
        for i in range(depth):
            if len(stride_q[i]) > 0:
                _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
            pool_kv_stride_size.append([i] + _stride_kv)
    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [(s + 1 if s > 1 else s) for s in pool_kv_stride_size[i][1:]]
    for i in range(depth):
        num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
        patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
        dim_out = round_width(patch_embed_dim, dim_mul[i + 1], divisor=round_width(num_heads, head_mul[i + 1]))
        mvit_blocks.append(MultiScaleBlock(dim=patch_embed_dim, dim_out=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, dropout_rate=dropout_rate_block, droppath_rate=dpr[i], norm_layer=block_norm_layer, kernel_q=pool_q[i], kernel_kv=pool_kv[i], stride_q=stride_q[i], stride_kv=stride_kv[i], pool_mode=pooling_mode, has_cls_embed=cls_embed_on, pool_first=pool_first))
    embed_dim = dim_out
    norm_embed = None if norm_layer is None else norm_layer(embed_dim)
    head_model = None
    return MultiscaleVisionTransformers(patch_embed=patch_embed, cls_positional_encoding=cls_positional_encoding, pos_drop=pos_drop if dropout_rate_block > 0.0 else None, norm_patch_embed=norm_patch_embed, blocks=mvit_blocks, norm_embed=norm_embed, head=head_model)


class MaskFeat(nn.Module):
    """
	Multiscale Vision Transformers
	Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
	https://arxiv.org/abs/2104.11227
	"""

    def __init__(self, img_size=224, num_frames=16, input_channels=3, feature_dim=10, patch_embed_dim=96, conv_patch_embed_kernel=(3, 7, 7), conv_patch_embed_stride=(2, 4, 4), conv_patch_embed_padding=(1, 3, 3), embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]], atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]], pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]], pool_kv_stride_adaptive=[1, 8, 8], pool_kvq_kernel=[3, 3, 3], head=None, pretrain_pth=None, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.stride = conv_patch_embed_stride
        self.downsample_rate = 2 ** len(pool_q_stride_size)
        self.embed_dims = 2 ** len(embed_dim_mul) * patch_embed_dim
        self.patch_embed = create_conv_patch_embed(in_channels=input_channels, out_channels=patch_embed_dim, conv_kernel_size=conv_patch_embed_kernel, conv_stride=conv_patch_embed_stride, conv_padding=conv_patch_embed_padding, conv=nn.Conv3d)
        self.mvit = create_multiscale_vision_transformers(spatial_size=img_size, temporal_size=num_frames, embed_dim_mul=embed_dim_mul, atten_head_mul=atten_head_mul, pool_q_stride_size=pool_q_stride_size, pool_kv_stride_adaptive=pool_kv_stride_adaptive, pool_kvq_kernel=pool_kvq_kernel, head=head)
        in_features = self.mvit.norm_embed.normalized_shape[0]
        out_features = feature_dim
        self.decoder_pred = nn.Linear(in_features, feature_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))
        w = self.patch_embed.patch_model.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if pretrain_pth is not None:
            self.init_weights(pretrain_pth)

    def init_weights(self, pretrain_pth):
        init_from_kinetics_pretrain_(self, pretrain_pth)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, target_x, mask, cube_marker, visualize=False):
        x = self.forward_features(x, mask)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        x = rearrange(x, 'b (t h w) (dt dc) -> b (t dt) h w dc', dt=self.stride[0], t=self.num_frames // self.stride[0], h=self.img_size // (self.stride[1] * self.downsample_rate), w=self.img_size // (self.stride[2] * self.downsample_rate))
        mask = repeat(mask, 'b t h w -> b (t dt) h w', dt=self.stride[0])
        center_index = torch.zeros(self.num_frames, device=mask.device)
        for i, mark_item in enumerate(cube_marker):
            for marker in mark_item:
                start_frame, span_frame = marker
                center_index[start_frame * self.stride[0] + span_frame * self.stride[0] // 2] = 1
            mask[i, ~center_index] = 0
            center_index.zero_()
        loss = (x - target_x) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-05)
        if visualize:
            mask_preds = x[:, center_index]
            mask_preds = rearrange(mask_preds, 'b t h w (dh dw c o) -> b t (h dh) (w dw) c o', dh=2, dw=2, c=3, o=9)
            return x, loss, mask_preds, center_index
        else:
            return x, loss

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x.transpose(1, 2))
        B, L, C = x.shape
        if mask is not None:
            mask_token = self.mask_token.expand(B, L, -1)
            dense_mask = repeat(mask, 'b t h w -> b t (h dh) (w dw)', dh=self.downsample_rate, dw=self.downsample_rate)
            w = dense_mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w
        x = self.mvit(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassificationHead,
     lambda: ([], {'num_classes': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FFNWithPreNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     True),
    (MultiheadAttentionWithPreNorm,
     lambda: ([], {'embed_dims': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_mx_mark_VideoTransformer_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

