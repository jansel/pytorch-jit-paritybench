import sys
_module = sys.modules[__name__]
del sys
conf = _module
generate_dir_structure = _module
setup = _module
tests = _module
test_attention = _module
test_decoder = _module
test_encoder = _module
test_models = _module
test_viz = _module
visualize = _module
vformer = _module
attention = _module
convvt = _module
cross = _module
gated_positional = _module
memory_efficient = _module
spatial = _module
vanilla = _module
window = _module
common = _module
base_model = _module
base_trainer = _module
blocks = _module
decoder = _module
mlp = _module
perceiver_io = _module
task_heads = _module
detection = _module
head = _module
segmentation = _module
head = _module
encoder = _module
convit = _module
convvt = _module
cross = _module
embedding = _module
convvt = _module
cvt = _module
linear = _module
overlappatch = _module
patch = _module
pos_embedding = _module
video_patch_embeddings = _module
nn = _module
perceiver_io = _module
pyramid = _module
swin = _module
vanilla = _module
vivit = _module
functional = _module
merge = _module
norm = _module
models = _module
classification = _module
cct = _module
convit = _module
convvt = _module
cross = _module
cvt = _module
perceiver_io = _module
pyramid = _module
swin = _module
vanilla = _module
visformer = _module
vivit = _module
PVT = _module
detection = _module
segmentation = _module
dense = _module
dpt = _module
utils = _module
dpt_utils = _module
registry = _module
window_utils = _module
viz = _module
vit_grad_rollout = _module
vit_rollout = _module

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


import torchvision.transforms.functional as F


from torchvision import io


from collections import OrderedDict


import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


from torchvision.transforms.functional import resize


from torchvision.ops import StochasticDepth


from torch import nn


import torch.utils.checkpoint as checkpoint


import types


import math


class Registry:
    """
    Class to register objects and then retrieve them by name.
    Parameters
    ----------
    name : str
        Name of the registry
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Method to register an object in the registry
        Parameters
        ----------
        obj : object, optional
            Object to register, defaults to None (which will return the decorator)
        name : str, optional
            Name of the object to register, defaults to None (which will use the name of the object)
        """
        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        """
        Method to retrieve an object from the registry
        Parameters
        ----------
        name : str
            Name of the object to retrieve
        Returns
        -------
        object
            Object registered under the given name
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def get_list(self):
        """
        Method to retrieve all objects from the registry
        Returns
        -------
        list
            List of all objects registered in the registry
        """
        return list(self._obj_map.keys())

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())


ATTENTION_REGISTRY = Registry('ATTENTION')


class ConvVTAttention(nn.Module):
    """
    Attention with Convolutional Projection introduced in Paper:  `Introducing Convolutions to Vision Transformers <https://arxiv.org/abs/2103.15808>`_

    Position-wise linear projection for Multi-Head Self-Attention (MHSA) replaced by Depth-wise separable convolutions

    Parameters
    -----------
    dim_in: int
        Dimension of input tensor
    dim_out: int
        Dimension of output tensor
    num_heads: int
        Number of heads in attention
    img_size: int
        Size of image
    attn_dropout: float
        Probability of dropout in attention
    proj_dropout: float
        Probability of dropout in convolution projection
    method: str
        Method of projection, ``'dw_bn'`` for depth-wise convolution and batch norm, ``'avg'`` for average pooling. default is ``'dw_bn'``
    kernel_size: int
        Size of kernel
    stride_kv: int
        Size of stride for key value
    stride_q: int
        Size of stride for query
    padding_kv: int
        Padding for key value
    padding_q: int
        Padding for query
    with_cls_token: bool
        Whether to include classification token, default is ```False```.
    """

    def __init__(self, dim_in, dim_out, num_heads, img_size, attn_dropout=0.0, proj_dropout=0.0, method='dw_bn', kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1, with_cls_token=False):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.with_cls_token = with_cls_token
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.h, self.w = img_size, img_size
        self.conv_proj_q = self._build_projection(dim_in, kernel_size, padding_q, stride_q, method)
        self.conv_proj_k = self._build_projection(dim_in, kernel_size, padding_kv, stride_kv, method)
        self.conv_proj_v = self._build_projection(dim_in, kernel_size, padding_kv, stride_kv, method)
        self.proj_q = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _build_projection(self, dim_in, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([('conv', nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in)), ('bn', nn.BatchNorm2d(dim_in))]))
        elif method == 'avg':
            proj = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True)
        else:
            raise ValueError('Unknown method ({})'.format(method))
        return proj

    def forward_conv(self, x):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, self.h * self.w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        q = self.conv_proj_q(x)
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = self.conv_proj_k(x)
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = self.conv_proj_v(x)
        v = rearrange(v, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        return q, k, v

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        q, k, v = self.forward_conv(x)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionWithClsToken(nn.Module):
    """
    Cross-Attention with Cls Token introduced in Paper: CrossViT: `Cross-Attention Multi-Scale Vision Transformer for Image Classification <https://arxiv.org/abs/2103.14899>`_

    In Cross-Attention, cls token from one branch and patch token from another branch are fused together.

    Parameters
    -----------
    cls_dim: int
        Dimension of cls token embedding
    patch_dim: int
        Dimension of patch token embeddings cls token to be fused with
    num_heads: int
        Number of cross-attention heads
    head_dim: int
        Dimension of each head

    """

    def __init__(self, cls_dim, patch_dim, num_heads=8, head_dim=64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.fl = nn.Linear(cls_dim, patch_dim) if not cls_dim == patch_dim else nn.Identity()
        self.gl = nn.Linear(patch_dim, cls_dim) if not cls_dim == patch_dim else nn.Identity()
        self.to_k = nn.Linear(patch_dim, inner_dim)
        self.to_v = nn.Linear(patch_dim, inner_dim)
        self.to_q = nn.Linear(patch_dim, inner_dim)
        self.cls_project = nn.Linear(inner_dim, patch_dim) if inner_dim != patch_dim else nn.Identity()
        self.attend = nn.Softmax(dim=-1)

    def forward(self, cls, patches):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        cls: torch.Tensor
            CLS token from one branch
        patch: torch.Tensor
            patch tokens from another branch
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying cross attention on input tensor

        """
        h = self.num_heads
        cls = self.fl(cls)
        x = torch.cat([cls, patches], dim=-2)
        q = self.to_q(cls)
        k = self.to_k(x)
        v = self.to_v(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, 'b h n d -> b n (h d)')
        attention_value = self.cls_project(attention_value)
        ycls = cls + attention_value
        ycls = self.gl(ycls)
        return ycls


class CrossAttention(nn.Module):
    """
    This variant of Cross Attention is iteratively used in Perciever IO.

    In Cross-Attention, cls token from one branch and patch token from another branch are fused together.

    Parameters
    ----------
    query_dim: int
        Dimension of query array
    context_dim: int
        Dimension of context array
    num_heads: int
        Number of cross-attention heads
    head_dim: int
        Dimension of each head

    """

    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, query_dim) if not inner_dim == query_dim else nn.Identity()
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x, context, mask=None):
        h = self.num_heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(attention.dtype).max
            mask = repeat(mask, 'b j -> b h () j', h=h)
            attention.masked_fill_(~mask, max_neg_value)
        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, 'b h n d -> b n (h d)')
        return self.to_out(attention_value)


class MemoryEfficientAttention(nn.Module):
    """
    Memory Effecient attention introduced in paper
    `Self-attention Does Not Need O(n2) Memory <https://arxiv.org/abs/2112.05682>`_

    Implementation based on `this repository <https://github.com/AminRezaei0x443/memory-efficient-attention>`_

    Parameters
    -----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    p_dropout: float
        Dropout Probability

    """

    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0.0, query_chunk_size=1024, key_chunk_size=4096):
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)
        self.num_heads = num_heads
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout)) if project_out else nn.Identity()

    @staticmethod
    def dynamic_slice(x, starts, sizes):
        starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
        for i, (start, size) in enumerate(zip(starts, sizes)):
            x = torch.index_select(x, i, torch.tensor(range(start, start + size), device=x.device))
        return x

    @staticmethod
    def summarize_chunk(query, key, value):
        attn_weights = torch.einsum('...qhd,...khd->...qhk', query, key)
        max_score, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('...vhf,...qhv->...qhf', value, exp_weights)
        max_score = torch.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(dim=-1), max_score

    @staticmethod
    def map_pt(f, xs):
        t = [f(x) for x in xs]
        return tuple(map(torch.stack, zip(*t)))

    @staticmethod
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, torch.stack(ys)

    def query_chunk_attention(self, query, key, value):
        num_kv, num_heads, k_features = key.shape[-3:]
        v_features = value.shape[-1]
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / k_features ** 0.5

        def chunk_scanner(chunk_idx):
            key_chunk = self.dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0), tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
            value_chunk = self.dynamic_slice(key, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0), tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))
            return checkpoint(self.summarize_chunk, query, key_chunk, value_chunk)
        chunk_values, chunk_weights, chunk_max = self.map_pt(chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))
        global_max, _ = torch.max(chunk_max, 0, keepdim=True)
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= torch.unsqueeze(max_diffs, -1)
        chunk_weights *= max_diffs
        all_values = chunk_values.sum(dim=0)
        all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
        return all_values / all_weights

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.num_heads), qkv)
        num_q, num_heads, q_features = q.shape[-3:]

        def inner_chunk_scanner(chunk_idx, _):
            query_chunk = self.dynamic_slice(q, tuple([0] * (q.ndim - 3)) + (chunk_idx, 0, 0), tuple(q.shape[:-3]) + (min(self.query_chunk_size, num_q), num_heads, q_features))
            return chunk_idx + self.query_chunk_size, self.query_chunk_attention(query_chunk, k, v)
        _, res = self.scan(inner_chunk_scanner, init=0, xs=None, length=int(np.ceil(num_q / self.query_chunk_size)))
        rl = [res[i] for i in range(res.shape[0])]
        att = torch.cat(rl, dim=-3)
        out = rearrange(att, 'b n h d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the embedding
    fn:nn.Module
        Attention class
    context_dim: int
        Dimension of the context array used in cross attention
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if context_dim is not None else None
        self.fn = fn

    def forward(self, x, **kwargs):
        if 'context' in kwargs.keys() and kwargs['context'] is not None:
            normed_context = self.context_norm(kwargs['context'])
            kwargs.update(context=normed_context)
        return self.fn(self.norm(x), **kwargs)


class SpatialAttention(nn.Module):
    """
    Spatial Reduction Attention introduced in : `Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions <https://arxiv.org/abs/2102.12122>`_
    This class also supports the linear complexity spatial attention in the improved `paper <https://arxiv.org/abs/2106.13797>`_

    Parameters
    -----------
    dim: int
        Dimension of the input tensor
    num_heads: int
        Number of attention heads
    sr_ratio :int
        Spatial Reduction ratio
    qkv_bias : bool
        If True, add a learnable bias to query, key, value, default is ``True``
    qk_scale : float, optional
        Override default qk scale of head_dim ** -0.5 if set
    attn_drop : float, optional
        Dropout rate
    proj_drop :float, optional
        Dropout rate
    linear : bool
        Whether to use linear Spatial attention,default is ``False``.
    activation : nn.Module
        Activation function, default is ``nn.GELU``.

    """

    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, linear=False, activation=nn.GELU):
        super(SpatialAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** 0.5
        inner_dim = head_dim * num_heads
        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)
        self.attn = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(p=attn_drop))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=proj_drop))
        self.linear = linear
        self.sr_ratio = sr_ratio
        self.norm = PreNorm(dim=dim, fn=activation() if linear else nn.Identity())
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, x, H, W):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height  of image patches
        W: int
            Width of image patches
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying spatial attention on input tensor

        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.norm(self.sr(x_).reshape(B, C, -1).permute(0, 2, 1))
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(x)


class VanillaSelfAttention(nn.Module):
    """
    Vanilla O(:math:`n^2`) Self attention introduced in `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Parameters
    -----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    p_dropout: float
        Dropout Probability

    """

    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0.0):
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout)) if project_out else nn.Identity()

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def pair(t):
    """
    Parameters
    ----------
    t: tuple[int] or int
    """
    return t if isinstance(t, tuple) else (t, t)


def get_relative_position_bias_index(window_size):
    """
    Parameters
    ----------
    window_size: int or tuple[int]
        Window size
    """
    window_size = pair(window_size)
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


class WindowAttention(nn.Module):
    """
    Implementation of Window Attention introduced in: `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_


    Parameters
    -----------
    dim: int
        Number of input channels.
    window_size : int or tuple[int]
        The height and width of the window.
    num_heads: int
        Number of attention heads.
    qkv_bias :bool
        If True, add a learnable bias to query, key, value, default is ``True``
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 if set
    attn_dropout: float, optional
        Dropout rate, default is 0.0.
    proj_dropout: float, optional
        Dropout rate, default is 0.0.

    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_dropout=0.0, proj_dropout=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = pair(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv_bias = True
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        relative_position_index = get_relative_position_bias_index(self.window_size)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out_1 = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(attn_dropout))
        self.to_out_2 = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.2)

    def forward(self, x, mask=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            input Tensor
        mask: torch.Tensor
            Attention mask used for shifted window attention, if None, window attention will be used,
            else attention mask will be taken into consideration.
            for better understanding you may refer `this github issue. <https://github.com/microsoft/Swin-Transformer/issues/38>`_

        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying Window-Attention or Shifted-Window-Attention on input tensor

        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.to_out_1(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.to_out_2(x)
        return x


class BaseClassificationModel(nn.Module):
    """

    Parameters
    -----------
    img_size: int
        Size of the image
    patch_size: int or tuple(int)
        Size of the patch
    in_channels: int
        Number of channels in input image
    pool: str
        Feature pooling type, must be one of {``mean``, ``cls``}
    """

    def __init__(self, img_size, patch_size, in_channels=3, pool='cls'):
        super(BaseClassificationModel, self).__init__()
        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert img_height % patch_height == 0 and img_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = img_height // patch_height * (img_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        assert pool in {'cls', 'mean'}, 'Feature pooling type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool


class DWConv(nn.Module):
    """
    Depth Wise Convolution

    Parameters
    -----------
    dim: int
        Dimension of the input tensor
    kernel_size_dwconv: int,optional
        Size of the convolution kernel, default is 3
    stride_dwconv: int
        Stride of the convolution, default is 1
    padding_dwconv: int or tuple or str
        Padding added to all sides of the input, default is 1
    bias_dwconv:bool
        Whether to add learnable bias to the output,default is ``True``.

    """

    def __init__(self, dim, kernel_size_dwconv=3, stride_dwconv=1, padding_dwconv=1, bias_dwconv=True):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size_dwconv, stride=stride_dwconv, padding=padding_dwconv, bias=bias_dwconv, groups=dim)

    def forward(self, x, H, W):
        """

        Parameters
        -----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of image patch
        W: int
            Width of image patch

        Returns
        --------
        torch.Tensor
            Returns output tensor after performing depth-wise convolution operation

        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


DECODER_REGISTRY = Registry('DECODER')


class MLPDecoder(nn.Module):
    """
    Parameters
    ----------
    config : int or tuple or list
        Configuration of the hidden layer(s)
    n_classes : int
        Number of classes for classification
    """

    def __init__(self, config=(1024,), n_classes=10):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.ModuleList()
        if not isinstance(config, list) and not isinstance(config, tuple):
            config = [config]
        if len(config) > 1:
            for i in range(len(config) - 1):
                self.decoder.append(nn.LayerNorm(config[i]))
                self.decoder.append(nn.Linear(config[i], config[i + 1]))
        self.decoder.append(nn.LayerNorm(config[-1]))
        self.decoder.append(nn.Linear(config[-1], n_classes))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor of size `n_classes`, Note that `torch.nn.Softmax` is not applied to the output tensor.

        """
        return self.decoder(x)


class FeedForward(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim: int, optional
        Dimension of the output tensor
    p_dropout: float
        Dropout probability, default=0.0

    """

    def __init__(self, dim, hidden_dim=None, out_dim=None, p_dropout=0.0):
        super().__init__()
        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(p_dropout), nn.Linear(hidden_dim, out_dim), nn.Dropout(p_dropout))

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------

        torch.Tensor
            Returns output tensor by performing linear operations and activation on input tensor

        """
        return self.net(x)


class PerceiverIODecoder(nn.Module):
    """
    Implementation of the Perceiver IO Decoder

    Parameters
    ----------
    dim: int
        Size of sequence to be encoded
    latent_dim: int
        Dimension of latent array
    queries_dim: int
        Dimension of queries array
    num_latents: int
        Number of latent arrays
    num_cross_heads: int
        Number of heads for cross attention
    cross_head_dim: int
        Dimension of cross attention head
    logits_dim: int, optional
        Dimension of output logits
    decoder_ff: bool
        Whether to include a feed forward layer for the decoder attention block
    """

    def __init__(self, dim=32, latent_dim=512, queries_dim=32, num_cross_heads=1, cross_head_dim=64, logits_dim=None, decoder_ff=False):
        super().__init__()
        self.decoder_cross_attn = PreNorm(queries_dim, CrossAttention(queries_dim, latent_dim, num_heads=num_cross_heads, head_dim=cross_head_dim), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if logits_dim is not None else nn.Identity()

    def forward(self, x, mask=None, queries=None):
        b, *_, device = *x.shape, x.device
        if queries is None:
            return x
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)
        latents = self.decoder_cross_attn(queries, context=x)
        if self.decoder_ff is not None:
            latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)


class DoubleConv(nn.Module):
    """
    Module consisting of two convolution layers and activations
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class SegmentationHead(nn.Module):
    """
    U-net like up-sampling block
    """

    def __init__(self, out_channels=1, embed_dims=[64, 128, 256, 512]):
        super(SegmentationHead, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in reversed(embed_dims):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(embed_dims[-1], embed_dims[-1] * 2)
        self.conv1 = nn.Conv2d(embed_dims[0], out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4)

    def forward(self, skip_connections):
        x = self.bottleneck(skip_connections[-1])
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x = self.conv1(x)
        return self.conv2(x)


class ConvEmbedding(nn.Module):
    """
    Projects image patches into embedding space using convolutional layer.

    Parameters
    -----------
    patch_size: int, default is 7
        Size of a patch
    in_channels: int, default is 3
        Number of input channels
    embedding_dim: int, default is 64
        Dimension of hidden layer
    stride: int or tuple, default is 4
        Stride of the convolution operation
    padding: int, default is 2
        Padding to all sides of the input
    """

    def __init__(self, patch_size=7, in_channels=3, embedding_dim=64, stride=4, padding=2):
        super().__init__()
        self.patch_size = patch_size, patch_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=self.patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Parameters
        -----------
        x: torch.tensor
            Input tensor

        Returns
        -----------
        torch.Tensor
            Returns output tensor (embedding) by applying a convolution operations on input tensor
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class ConvVTBlock(nn.Module):
    """
    Implementation of a Attention MLP block in CVT

    Parameters
    -----------
    dim_in: int
        Input dimensions
    dim_out: int
        Output dimensions
    num_heads: int
        Number of heads in attention
    img_size: int
        Size of image
    mlp_ratio: float
        Feature dimension expansion ratio in MLP, default is 4.
    p_dropout: float
        Probability of dropout in MLP, default is 0.0
    attn_dropout: float
        Probability of dropout in attention, default is 0.0
    drop_path: float
        Probability of droppath, default is 0.0
    with_cls_token: bool
        Whether to include classification token, default is False
    """

    def __init__(self, dim_in, dim_out, mlp_ratio=4.0, p_dropout=0.0, drop_path=0.0, drop_path_mode='batch', **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_in)
        self.attn = ConvVTAttention(dim_in, dim_out, **kwargs)
        self.drop_path = StochasticDepth(p=drop_path, mode=drop_path_mode) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_out)
        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = FeedForward(dim=dim_out, hidden_dim=dim_mlp_hidden, p_dropout=p_dropout)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn = self.attn(x)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


ENCODER_REGISTRY = Registry('ENCODER')


class ConvVTStage(nn.Module):
    """
    Implementation of a Stage in CVT

    Parameters
    -----------
    patch_size: int
        Size of patch, default is 16
    patch_stride: int
        Stride of patch, default is 4
    patch_padding: int
        Padding for patch, default is 0
    in_channels:int
        Number of input channels in image, default is 3
    img_size: int
        Size of the image, default is 224
    embedding_dim: int
        Embedding dimensions, default is 64
    depth: int
        Number of CVT Attention blocks in each stage, default is 1
    num_heads: int
        Number of heads in attention, default is 6
    mlp_ratio: float
        Feature dimension expansion ratio in MLP, default is 4.0
    p_dropout: float
        Probability of dropout in MLP, default is 0.0
    attn_dropout: float
        Probability of dropout in attention, default is 0.0
    drop_path_rate: float
        Probability for droppath, default is 0.0
    with_cls_token: bool
        Whether to include classification token, default is False
    kernel_size: int
        Size of kernel, default is 3
    padding_q: int
        Size of padding in q, default is 1
    padding_kv: int
        Size of padding in kv, default is 2
    stride_kv: int
        Stride in kv, default is 2
    stride_q: int
        Stride in q, default is 1
    init: str
        Initialization method, one of  {``trunc_norm`` or ``xavier``} default is ``trunc_norm``
    """

    def __init__(self, patch_size=7, patch_stride=4, patch_padding=0, in_channels=3, embedding_dim=64, depth=1, p_dropout=0.0, drop_path_rate=0.0, with_cls_token=False, init='trunc_norm', **kwargs):
        super().__init__()
        self.patch_embed = ConvEmbedding(patch_size=patch_size, in_channels=in_channels, embedding_dim=embedding_dim, stride=patch_stride, padding=patch_padding)
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.cls_token = None
        self.pos_drop = nn.Dropout(p=p_dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for j in range(depth):
            blocks.append(ConvVTBlock(dim_in=embedding_dim, dim_out=embedding_dim, p_dropout=p_dropout, with_cls_token=with_cls_token, drop_path=dpr[j], **kwargs))
        self.blocks = nn.ModuleList(blocks)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        elif init == 'trunc_norm':
            self.apply(self._init_weights_trunc_normal)
        else:
            raise ValueError('Init method {} not found'.format(init))

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x, cls_tokens


class VanillaEncoder(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the embedding
    depth: int
        Number of self-attention layers
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    mlp_dim: int
        Dimension of the hidden layer in the feed-forward layer
    p_dropout: float
        Dropout Probability
    attn_dropout: float
        Dropout Probability
    drop_path_rate: float
        Stochastic drop path rate
    """

    def __init__(self, embedding_dim, depth, num_heads, head_dim, mlp_dim, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, drop_path_mode='batch'):
        super().__init__()
        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(nn.ModuleList([PreNorm(dim=embedding_dim, fn=VanillaSelfAttention(dim=embedding_dim, num_heads=num_heads, head_dim=head_dim, p_dropout=attn_dropout)), PreNorm(dim=embedding_dim, fn=FeedForward(dim=embedding_dim, hidden_dim=mlp_dim, p_dropout=p_dropout))]))
        self.drop_path = StochasticDepth(p=drop_path_rate, mode=drop_path_mode) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor
        """
        for attn, ff in self.encoder:
            x = attn(x) + x
            x = self.drop_path(ff(x)) + x
        return x


class CrossEncoder(nn.Module):
    """
    Encoder block used in Cross-VIT .

    Parameters
    ----------
    embedding_dim_s : int
        Dimension of the embedding of smaller patches, default is 1024
    embedding_dim_l : int
        Dimension of the embedding of larger patches, default is 1024
    attn_heads_s : int
        Number of self-attention heads for the smaller patches, default is 16
    attn_heads_l : int
        Number of self-attention heads for the larger patches, default is 16
    cross_head_s : int
        Number of cross-attention heads for the smaller patches, default is 8
    cross_head_l : int
        Number of cross-attention heads for the larger patches, default is 8
    head_dim_s : int
        Dimension of the head of the attention for the smaller patches, default is 64
    head_dim_l : int
        Dimension of the head of the attention for the larger patches, default is 64
    cross_dim_head_s : int
        Dimension of the head of the cross-attention for the smaller patches, default is 64
    cross_dim_head_l : int
        Dimension of the head of the cross-attention for the larger patches, default is 64
    depth_s : int
        Number of self-attention layers in encoder for the smaller patches, default is 6
    depth_l : int
        Number of self-attention layers in encoder for the larger patches, default is 6
    mlp_dim_s : int
        Dimension of the hidden layer in the feed-forward layer for the smaller patches, default is 2048
    mlp_dim_l : int
        Dimension of the hidden layer in the feed-forward layer for the larger patches, default is 2048
    p_dropout_s : float
        Dropout probability for the smaller patches, default is 0.0
    p_dropout_l : float
        Dropout probability for the larger patches, default is 0.0
    """

    def __init__(self, embedding_dim_s=1024, embedding_dim_l=1024, attn_heads_s=16, attn_heads_l=16, cross_head_s=8, cross_head_l=8, head_dim_s=64, head_dim_l=64, cross_dim_head_s=64, cross_dim_head_l=64, depth_s=6, depth_l=6, mlp_dim_s=2048, mlp_dim_l=2048, p_dropout_s=0.0, p_dropout_l=0.0):
        super().__init__()
        self.s = VanillaEncoder(embedding_dim_s, depth_s, attn_heads_s, head_dim_s, mlp_dim_s, p_dropout_s)
        self.l = VanillaEncoder(embedding_dim_l, depth_l, attn_heads_l, head_dim_l, mlp_dim_l, p_dropout_l)
        self.attend_s = CrossAttentionWithClsToken(embedding_dim_s, embedding_dim_l, cross_head_s, cross_dim_head_s)
        self.attend_l = CrossAttentionWithClsToken(embedding_dim_l, embedding_dim_s, cross_head_l, cross_dim_head_l)

    def forward(self, emb_s, emb_l):
        emb_s = self.s(emb_s)
        emb_l = self.l(emb_l)
        s_cls, s_patches = (lambda t: (t[:, 0:1, :], t[:, 1:, :]))(emb_s)
        l_cls, l_patches = (lambda t: (t[:, 0:1, :], t[:, 1:, :]))(emb_l)
        s_cls = self.attend_s(s_cls, l_patches)
        l_cls = self.attend_l(l_cls, s_patches)
        emb_l = torch.cat([l_cls, l_patches], dim=1)
        emb_s = torch.cat([s_cls, s_patches], dim=1)
        return emb_s, emb_l


class CVTEmbedding(nn.Module):
    """
    Projects image patches into embedding space using multiple Convolution and maxpooling layers.

    Parameters
    -----------
    kernel_size: int or tuple
        Size of the kernel used in convolution
    stride: int or tuple
        Stride of the convolution operation
    padding: int
        Padding to all sides of the input
    pooling_kernel_size: int or tuple(int)
        Size of the kernel used in  MaxPool2D,default is 3
    pooling_stride: int or tuple(int)
        Size of the stride in MaxPool2D, default is 2
    pooling_padding: int
        padding in the MaxPool2D
    num_conv_layers: int
        Number of Convolution layers in the encoder,default is 1
    in_channels: int
        Number of input channels in image, default is 3
    out_channels: int
        Number of output channels, default is 64
    in_planes: int
        This will be number of channels in the self.conv_layer's convolution except 1st layer and last layer.
    activation: nn.Module, optional
        Activation Layer, default is None
    max_pool: bool
        Whether to have max-pooling or not, change this parameter to False when using in CVT model, default is True
    conv_bias:bool
        Whether to add learnable bias in the convolution operation,default is False
    """

    def __init__(self, kernel_size, stride, padding, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, num_conv_layers=1, in_channels=3, out_channels=64, in_planes=64, activation=None, max_pool=True, conv_bias=False):
        super(CVTEmbedding, self).__init__()
        n_filter_list = [in_channels] + [in_planes for _ in range(num_conv_layers - 1)] + [out_channels]
        self.conv_layers = nn.ModuleList([])
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.ModuleList([nn.Conv2d(n_filter_list[i], n_filter_list[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias), nn.Identity() if activation is None else activation(), nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding) if max_pool else nn.Identity()]))
        self.flatten = nn.Flatten(2, 3)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.tensor
            Input tensor

        Returns
        -----------
        torch.Tensor
            Returns output tensor (embedding) by applying multiple convolution and max-pooling operations on input tensor

        """
        for conv2d, activation, maxpool in self.conv_layers:
            x = maxpool(activation(conv2d(x)))
            return self.flatten(x).transpose(-2, -1)


class LinearEmbedding(nn.Module):
    """
    Projects image patches into embedding space using Linear layer.

    Parameters
    -----------
    embedding_dim: int
        Dimension of the resultant embedding
    patch_height: int
        Height of the patch
    patch_width: int
        Width of the patch
    patch_dim: int
        Dimension of the patch

    """

    def __init__(self, embedding_dim, patch_height, patch_width, patch_dim):
        super().__init__()
        self.patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.Linear(patch_dim, embedding_dim))

    def forward(self, x):
        """

        Parameters
        -----------
        x: torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns patch embeddings of size `embedding_dim`

        """
        return self.patch_embedding(x)


class OverlapPatchEmbed(nn.Module):
    """

    Parameters
    ----------
    img_size: int
        Image Size
    patch_size: int or tuple(int)
        Patch Size
    stride: int
        Stride of the convolution, default is 4
    in_channels: int
        Number of input channels in the image, default is 3
    embedding_dim: int
        Number of linear projection output channels,default is 768
    norm_layer: nn.Module, optional
        Normalization layer, default is nn.LayerNorm

    """

    def __init__(self, img_size, patch_size, stride=4, in_channels=3, embedding_dim=768, norm_layer=nn.LayerNorm):
        super(OverlapPatchEmbed, self).__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = norm_layer(embedding_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
              Input tensor

        Returns
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of Patch
        W: int
            Width of Patch

        """
        x = self.proj(x)
        H, W = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


class PatchEmbedding(nn.Module):
    """

    Parameters
    ----------
    img_size: int
        Image Size
    patch_size: int
        Patch Size
    in_channels: int
        Number of input channels in the image
    embedding_dim: int
        Number of linear projection output channels
    norm_layer: nn.Module,
        Normalization layer, Default is `nn.LayerNorm`

    """

    def __init__(self, img_size, patch_size, in_channels, embedding_dim, norm_layer=nn.LayerNorm):
        super(PatchEmbedding, self).__init__()
        self.img_size = pair(img_size)
        self.patch_size = pair(patch_size)
        self.patch_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x:torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying convolution operation with same `kernel_size` and `stride` on input tensor.

        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f'Input Image Size {H}*{W} doesnt match model {self.img_size[0]}*{self.img_size[1]}'
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PVTPosEmbedding(nn.Module):
    """
    Positional Embedding class used in Pyramid vision transformer.

    Parameters
    -----------
    pos_shape : int or tuple(int)
        The shape of the absolute position embedding.
    pos_dim : int
        The dimension of the absolute position embedding.
    p_dropout : float, optional
        Probability of an element to be zeroed, default is 0.2
    std: float
        Standard deviation for truncated normal distribution
    """

    def __init__(self, pos_shape, pos_dim, p_dropout=0.0, std=0.02):
        super().__init__()
        pos_shape = pair(pos_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim
        self.drop = nn.Dropout(p=p_dropout)
        nn.init.trunc_normal_(self.pos_embed, std=std)

    def resize_pos_embed(self, pos_embed, shape, mode='bilinear', **kwargs):
        """
        Parameters
        -----------
            pos_embed : torch.Tensor
                Position embedding weights
            shape : tuple
                Required shape
            mode : str  (``nearest`` | ``linear`` | ``bilinear`` | ``bicubic`` | ``trilinear`` )
                Algorithm used for up/down sampling, default is ``bilinear``.
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, -1 * pos_h * pos_w:]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(pos_embed_weight, size=shape, mode=mode, **kwargs)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight
        return pos_embed

    def forward(self, x, H, W, mode='bilinear'):
        try:
            x = x + self.pos_embed
        except:
            x = x + self.resize_pos_embed(self.pos_embed, (H, W), mode)
        return self.drop(x)


class PosEmbedding(nn.Module):
    """
    Generalised Positional Embedding class
    """

    def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
        super(PosEmbedding, self).__init__()
        if not sinusoidal:
            if isinstance(shape, int):
                shape = [1, shape, dim]
            else:
                shape = [1] + list(shape) + [dim]
            self.pos_embed = nn.Parameter(torch.zeros(shape))
        else:
            pe = torch.FloatTensor([[(p / 10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(shape)])
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            self.pos_embed = pe
            self.pos_embed.requires_grad = False
        nn.init.trunc_normal_(self.pos_embed, std=std)
        self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed
        return self.pos_drop(x)


class LinearVideoEmbedding(nn.Module):
    """

    Parameters
    -----------
    embedding_dim: int
        Dimension of the resultant embedding
    patch_height: int
        Height of the patch
    patch_width: int
        Width of the patch
    patch_dim: int
        patch_dimension

    """

    def __init__(self, embedding_dim, patch_height, patch_width, patch_dim):
        super().__init__()
        self.patch_embedding = nn.Sequential(Rearrange('b t c (h ph) (w pw) -> b t (h w) (ph pw c)', ph=patch_height, pw=patch_width), nn.Linear(patch_dim, embedding_dim))

    def forward(self, x):
        """

        Parameters
        -----------
        x: torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns patch embeddings of size `embedding_dim`

        """
        return self.patch_embedding(x)


class TubeletEmbedding(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the resultant embedding
    tubelet_t: int
        Temporal length of single tube/patch
    tubelet_h: int
        Heigth  of single tube/patch
    tubelet_w: int
        Width of single tube/patch
    in_channels: int
        Number of channels
    """

    def __init__(self, embedding_dim, tubelet_t, tubelet_h, tubelet_w, in_channels):
        super(TubeletEmbedding, self).__init__()
        tubelet_dim = in_channels * tubelet_h * tubelet_w * tubelet_t
        self.tubelet_embedding = nn.Sequential(Rearrange('b  (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=tubelet_t, ph=tubelet_h, pw=tubelet_w), nn.Linear(tubelet_dim, embedding_dim))

    def forward(self, x):
        """

        Parameters
        ----------
        x: Torch.tensor
            Input tensor

        """
        return self.tubelet_embedding(x)


class PerceiverIOEncoder(nn.Module):
    """
    Implementation of the Perceiver IO Encoder containing Iterative Cross Attention and Processor

    Parameters
    ----------
    dim: int
        Size of sequence to be encoded
    depth: int
        Depth of latent attention blocks
    latent_dim: int
        Dimension of latent array
    num_latents: int
        Number of latent arrays
    num_cross_heads: int
        Number of heads for cross attention
    num_latent_heads: int
        Number of heads for latent attention
    cross_head_dim: int
        Dimension of cross attention head
    latent_head_dim: int
        Dimension of latent attention head
    """

    def __init__(self, dim=32, depth=6, latent_dim=512, num_latents=512, num_cross_heads=1, num_latent_heads=8, cross_head_dim=64, latent_head_dim=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = PreNorm(latent_dim, CrossAttention(latent_dim, dim, num_heads=num_cross_heads, head_dim=cross_head_dim), context_dim=dim)
        self.cross_ff = PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = VanillaSelfAttention(latent_dim, num_heads=num_latent_heads, head_dim=latent_head_dim)
        get_latent_ff = PreNorm(latent_dim, FeedForward(latent_dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn, get_latent_ff]))

    def forward(self, x, mask=None):
        b, *_, device = *x.shape, x.device
        inner_x = repeat(self.latents, 'n d -> b n d', b=b)
        inner_x = self.cross_attn(inner_x, context=x, mask=mask) + inner_x
        inner_x = self.cross_ff(inner_x) + inner_x
        for self_attn, self_ff in self.layers:
            inner_x = self_attn(inner_x) + inner_x
            inner_x = self_ff(inner_x) + inner_x
        return inner_x


class PVTFeedForward(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim:int, optional
        Dimension of output tensor
    activation: nn.Module
        Activation Layer, default is nn.GELU
    p_dropout: float
        Dropout probability/rate, default is 0.0
    linear: bool
        Whether to use linear Spatial attention,default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is False

    kernel_size_dwconv: int
        `kernel_size` parameter for 2D convolution used in Depth wise convolution
    stride_dwconv: int
        `stride` parameter for 2D convolution used in Depth wise convolution
    padding_dwconv: int
        `padding` parameter for 2D convolution used in Depth wise convolution
    bias_dwconv:bool
        `bias` parameter for 2D convolution used in Depth wise convolution
    """

    def __init__(self, dim, hidden_dim=None, out_dim=None, activation=nn.GELU, p_dropout=0.0, linear=False, use_dwconv=False, **kwargs):
        super(PVTFeedForward, self).__init__()
        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.use_dwconv = use_dwconv
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True) if linear else nn.Identity()
        if use_dwconv:
            self.dw_conv = DWConv(dim=hidden_dim, **kwargs)
        self.to_out = nn.Sequential(activation(), nn.Dropout(p=p_dropout), nn.Linear(hidden_dim, out_dim), nn.Dropout(p=p_dropout))

    def forward(self, x, **kwargs):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of image patch
        W: int
            Width of image patch

        Returns
        --------
        torch.Tensor
            Returns output tensor

        """
        x = self.relu(self.fc1(x))
        if self.use_dwconv:
            x = self.dw_conv(x, **kwargs)
        return self.to_out(x)


class PVTEncoder(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    num_heads: int
        Number of attention heads
    mlp_ratio:
        Ratio of MLP hidden dimension to embedding dimension
    depth: int
        Number of attention layers in the encoder
    qkv_bias: bool
        Whether to add a bias vector to the q,k, and v matrices
    qk_scale:float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float
        Dropout probability
    attn_dropout: float
        Dropout probability
    drop_path: tuple(float)
        List of stochastic drop rate
    activation: nn.Module
        Activation layer
    use_dwconv:bool
        Whether to use depth-wise convolutions in overlap-patch embedding
    sr_ratio: float
        Spatial Reduction ratio
    linear: bool
        Whether to use linear Spatial attention, default is ```False```.
    drop_path_mode: str
        Mode for `StochasticDepth <https://pytorch.org/vision/main/generated/torchvision.ops.stochastic_depth.html>_ , must be one of {``batch`` or ``row``}
    """

    def __init__(self, dim, num_heads, mlp_ratio, depth, qkv_bias, qk_scale, p_dropout, attn_dropout, drop_path, activation, use_dwconv, sr_ratio, linear=False, drop_path_mode='batch'):
        super(PVTEncoder, self).__init__()
        self.encoder = nn.ModuleList([])
        for i in range(depth):
            self.encoder.append(nn.ModuleList([PreNorm(dim=dim, fn=SpatialAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_dropout, proj_drop=p_dropout, sr_ratio=sr_ratio, linear=linear)), PreNorm(dim=dim, fn=PVTFeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), activation=activation, p_dropout=p_dropout, linear=linear, use_dwconv=use_dwconv))]))
            self.drop_path = StochasticDepth(p=drop_path[i], mode=drop_path_mode) if drop_path[i] > 0.0 else nn.Identity()

    def forward(self, x, **kwargs):
        for prenorm_attn, prenorm_ff in self.encoder:
            x = x + self.drop_path(prenorm_attn(x, **kwargs))
            x = x + self.drop_path(prenorm_ff(x, **kwargs))
        return x


def window_partition(x, window_size):
    """
    Parameters
    -----------
    x: torch.Tensor
        input tensor
    window_size: int
        window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def create_mask(window_size, shift_size, H, W):
    """
    Parameters
    ----------
    window_size: int
        Window Size
    shift_size: int
        Shift_size

    """
    img_mask = torch.zeros(1, H, W, 1)
    h_slices = slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)
    w_slices = slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def cyclicshift(input, shift_size, dims=None):
    """
    Parameters
    -----------
    input: torch.Tensor
        input tensor
    shift_size: int or tuple(int)
        Number of places by which input tensor is shifted
    dims: int or tuple(int),optional
        Axis along which to roll
    """
    return torch.roll(input, shifts=pair(shift_size), dims=(1, 2) if dims == None else dims)


def window_reverse(windows, window_size, H, W):
    """
    Parameters
    -----------
    windows: torch.Tensor
    window_size: int
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinEncoderBlock(nn.Module):
    """

    Parameters
    -----------
    dim: int
        Number of the input channels
    input_resolution: int or tuple[int]
        Input resolution of patches
    num_heads: int
        Number of attention heads
    window_size: int
        Window size
    shift_size: int
        Shift size for Shifted Window Masked Self Attention (SW_MSA)
    mlp_ratio: float
        Ratio of MLP hidden dimension to embedding dimension
    qkv_bias: bool, default= True
        Whether to add a bias vector to the q,k, and v matrices
    qk_scale: float, Optional

    p_dropout: float
        Dropout rate
    attn_dropout: float
        Dropout rate
    drop_path_rate: float
        Stochastic depth rate
    norm_layer:nn.Module
        Normalization layer, default is `nn.LayerNorm`

    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, drop_path_mode='batch'):
        super(SwinEncoderBlock, self).__init__()
        self.dim = dim
        self.input_resolution = pair(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        hidden_dim = int(dim * mlp_ratio)
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < window_size, 'shift size must range from 0 to window size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim=dim, window_size=pair(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_dropout=attn_dropout, proj_dropout=p_dropout)
        self.drop_path = StochasticDepth(p=drop_path_rate, mode=drop_path_mode) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, p_dropout=p_dropout)
        if self.shift_size > 0:
            attn_mask = create_mask(self.window_size, self.shift_size, H=self.input_resolution[0], W=self.input_resolution[1])
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor

        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'Input tensor shape not compatible'
        skip_connection = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = cyclicshift(x, shift_size=-self.shift_size)
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = cyclicshift(shifted_x, shift_size=self.shift_size).view(B, H * W, C)
        else:
            x = shifted_x.view(B, H * W, C)
        x = skip_connection + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinEncoder(nn.Module):
    """

    Parameters
    -----------
    dim: int
        Number of input channels.
    input_resolution: tuple[int]
        Input resolution.
    depth: int
        Number of blocks.
    num_heads: int
        Number of attention heads.
    window_size: int
        Local window size.
    mlp_ratio: float
        Ratio of MLP hidden dim to embedding dim.
    qkv_bias: bool, default is True
       Whether to add a bias vector to the q,k, and v matrices
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Window Attention if set
    p_dropout: float,
        Dropout rate.
    attn_dropout: float, optional
        Attention dropout rate
    drop_path_rate: float or tuple[float]
        Stochastic depth rate.
    norm_layer: nn.Module
        Normalization layer. default is nn.LayerNorm
    downsample: nn.Module, optional
        Downsample layer(like PatchMerging) at the end of the layer, default is None

    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qkv_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super(SwinEncoder, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinEncoderBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qkv_scale, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor

        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ViViTEncoderBlock(nn.Module):
    """For model 3 only"""

    def __init__(self, dim, num_heads, head_dim, p_dropout, out_dim=None, hidden_dim=None):
        super(ViViTEncoderBlock, self).__init__()
        self.temporal_attention = PreNorm(dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout))
        self.spatial_attention = PreNorm(dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout))
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):
        b, n, s, d = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.spatial_attention(x) + x
        x = x.reshape(b, n, s, d).transpose(1, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.temporal_attention(x) + x
        x = self.mlp(x) + x
        x = x.reshape(b, n, s, d)
        return x


class ViViTEncoder(nn.Module):
    """model 3 only"""

    def __init__(self, dim, num_heads, head_dim, p_dropout, depth, out_dim=None, hidden_dim=None):
        super(ViViTEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for _ in range(depth):
            self.encoder.append(ViViTEncoderBlock(dim, num_heads, head_dim, p_dropout, out_dim, hidden_dim))

    def forward(self, x):
        b = x.shape[0]
        for blk in self.encoder:
            x = blk(x)
        x = x.reshape(b, -1, x.shape[-1])
        return x


class PatchMerging(nn.Module):
    """

    Parameters
    ----------
    input_resolution: int or tuple[int]
        Resolution of input features
    dim : int
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = pair(input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MODEL_REGISTRY = Registry('MODEL')


class CCT(BaseClassificationModel):
    """
    Implementation of `Escaping the Big Data Paradigm with Compact Transformers <https://arxiv.org/abs/2104.05704>`_

    Parameters
    -----------
    img_size: int
        Size of the image
    patch_size: int
        Size of the single patch in the image
    in_channels: int
        Number of input channels in image
    seq_pool:bool
        Whether to use sequence pooling or not
    embedding_dim: int
        Patch embedding dimension
    num_layers: int
        Number of Encoders in encoder block
    num_heads: int
        Number of heads in each transformer layer
    mlp_ratio:float
        Ratio of mlp heads to embedding dimension
    n_classes: int
        Number of classes for classification
    p_dropout: float
        Dropout probability
    attn_dropout: float
        Dropout probability
    drop_path: float
        Stochastic depth rate, default is 0.1
    positional_embedding: str
        One of the string values {``'learnable'``, ``'sine'`` , ``None``}, default is ``'learnable'``.
    decoder_config: tuple(int) or int
        Configuration of the decoder. If None, the default configuration is used.
    pooling_kernel_size: int or tuple(int)
        Size of the kernel in MaxPooling operation
    pooling_stride: int or tuple(int)
        Stride of MaxPooling operation
    pooling_padding: int
        Padding in MaxPooling operation
    """

    def __init__(self, img_size=224, patch_size=4, in_channels=3, seq_pool=True, embedding_dim=768, num_layers=1, head_dim=96, num_heads=1, mlp_ratio=4.0, n_classes=1000, p_dropout=0.1, attn_dropout=0.1, drop_path=0.1, positional_embedding='learnable', decoder_config=(768, 1024), pooling_kernel_size=3, pooling_stride=2, pooling_padding=1):
        super().__init__(img_size=img_size, patch_size=patch_size)
        assert img_size % patch_size == 0, f'Image size ({img_size}) has to be divisible by patch size ({patch_size})'
        img_size = pair(img_size)
        self.in_channels = in_channels
        self.embedding = CVTEmbedding(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0, max_pool=True, pooling_kernel_size=pooling_kernel_size, pooling_stride=pooling_stride, pooling_padding=pooling_padding, activation=nn.ReLU, num_conv_layers=1, conv_bias=True)
        positional_embedding = positional_embedding if positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = self.embedding.sequence_length(n_channels=in_channels, height=img_size[0], width=img_size[1])
        self.seq_pool = seq_pool
        assert self.sequence_length is not None or positional_embedding == 'none', f'Positional embedding is set to {positional_embedding} and the sequence length was not specified.'
        if not seq_pool:
            self.sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
        if positional_embedding != 'none':
            self.positional_emb = PosEmbedding(self.sequence_length, dim=embedding_dim, drop=p_dropout, sinusoidal=True if positional_embedding == 'sine' else False)
        else:
            self.positional_emb = None
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.encoder_blocks = nn.ModuleList([VanillaEncoder(embedding_dim=embedding_dim, num_heads=num_heads, depth=1, head_dim=head_dim, mlp_dim=hidden_dim, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=dpr[i]) for i in range(num_layers)])
        if decoder_config is not None:
            if not isinstance(decoder_config, list) and not isinstance(decoder_config, tuple):
                decoder_config = [decoder_config]
            assert decoder_config[0] == embedding_dim, f'Configurations do not match for MLPDecoder, First element of `decoder_config` expected to be {embedding_dim}, got {decoder_config[0]} '
            self.decoder = MLPDecoder(config=decoder_config, n_classes=n_classes)
        else:
            self.decoder = MLPDecoder(config=embedding_dim, n_classes=n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        x = self.embedding(x)
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.in_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if self.positional_emb is not None:
            x = self.positional_emb(x)
        for blk in self.encoder_blocks:
            x = blk(x)
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        x = self.decoder(x)
        return x


class ConvVT(nn.Module):
    """
    Implementation of `CvT: Introducing Convolutions to Vision Transformers <https://arxiv.org/abs/2103.15808>`_

    Parameters
    -----------
    img_size: int
        Size of the image, default is 224
    in_channels:int
        Number of input channels in image, default is 3
    num_stages: int
        Number of stages in encoder block, default is 3
    n_classes: int
        Number of classes for classification, default is 1000
    * The following are all in list of int/float with length num_stages
    patch_size: list[int]
        Size of patch, default is [7, 3, 3]
    patch_stride: list[int]
        Stride of patch, default is [4, 2, 2]
    patch_padding: list[int]
        Padding for patch, default is [2, 1, 1]
    embedding_dim: list[int]
        Embedding dimensions, default is [64, 192, 384]
    depth: list[int]
        Number of CVT Attention blocks in each stage, default is [1, 2, 10]
    num_heads: list[int]
        Number of heads in attention, default is [1, 3, 6]
    mlp_ratio: list[float]
        Feature dimension expansion ratio in MLP, default is [4.0, 4.0, 4.0]
    p_dropout: list[float]
        Probability of dropout in MLP, default is [0, 0, 0]
    attn_dropout: list[float]
        Probability of dropout in attention, default is [0, 0, 0]
    drop_path_rate: list[float]
        Probability for droppath, default is [0, 0, 0.1]
    kernel_size: list[int]
        Size of kernel, default is [3, 3, 3]
    padding_q: list[int]
        Size of padding in q, default is [1, 1, 1]
    padding_kv: list[int]
        Size of padding in kv, default is [2, 2, 2]
    stride_kv: list[int]
        Stride in kv, default is [2, 2, 2]
    stride_q: list[int]
        Stride in q, default is [1, 1, 1]
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3], patch_stride=[4, 2, 2], patch_padding=[2, 1, 1], embedding_dim=[64, 192, 384], num_heads=[1, 3, 6], depth=[1, 2, 10], mlp_ratio=[4.0, 4.0, 4.0], p_dropout=[0, 0, 0], attn_dropout=[0, 0, 0], drop_path_rate=[0, 0, 0.1], kernel_size=[3, 3, 3], padding_q=[1, 1, 1], padding_kv=[1, 1, 1], stride_kv=[2, 2, 2], stride_q=[1, 1, 1], in_channels=3, num_stages=3, n_classes=1000):
        super().__init__()
        self.n_classes = n_classes
        self.num_stages = num_stages
        self.stages = nn.ModuleList([])
        for i in range(self.num_stages):
            stage = ConvVTStage(in_channels=in_channels, img_size=img_size // (4 * 2 ** i), with_cls_token=False if i < self.num_stages - 1 else True, patch_size=patch_size[i], patch_stride=patch_stride[i], patch_padding=patch_padding[i], embedding_dim=embedding_dim[i], depth=depth[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], p_dropout=p_dropout[i], attn_dropout=attn_dropout[i], drop_path_rate=drop_path_rate[i], kernel_size=kernel_size[i], padding_q=padding_q[i], padding_kv=padding_kv[i], stride_kv=stride_kv[i], stride_q=stride_q[i])
            self.stages.append(stage)
            in_channels = embedding_dim[i]
        self.norm = nn.LayerNorm(embedding_dim[-1])
        self.head = nn.Linear(embedding_dim[-1], n_classes) if n_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = self.stages[i](x)
        x = self.norm(cls_tokens)
        x = torch.squeeze(x)
        x = self.head(x)
        return x


class _cross_p(BaseClassificationModel):

    def __init__(self, img_size, patch_size, latent_dim=1024, in_channels=3, p_dropout_embedding=0.0):
        super().__init__(img_size, patch_size, in_channels)
        self.patch_embedding = LinearEmbedding(latent_dim, self.patch_height, self.patch_width, self.patch_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, latent_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.embedding_dropout = nn.Dropout(p_dropout_embedding)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.embedding_dropout(x)
        return x


class CrossViT(BaseClassificationModel):
    """
    Implementation of `CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification <https://arxiv.org/abs/2103.14899>`_

    Parameters
    -----------
    img_size: int
        Size of the image
    patch_size_s: int
        Size of the smaller patches
    patch_size_l: int
        Size of the larger patches
    n_classes: int
        Number of classes for classification
    cross_dim_head_s: int
        Dimension of the head of the cross-attention for the smaller patches
    cross_dim_head_l: int
        Dimension of the head of the cross-attention for the larger patches
    latent_dim_s: int
        Dimension of the hidden layer for the smaller patches
    latent_dim_l: int
        Dimension of the hidden layer for the larger patches
    head_dim_s: int
        Dimension of the head of the attention for the smaller patches
    head_dim_l: int
        Dimension of the head of the attention for the larger patches
    depth_s: int
        Number of attention layers in encoder for the smaller patches
    depth_l: int
        Number of attention layers in encoder for the larger patches
    attn_heads_s: int
        Number of attention heads for the smaller patches
    attn_heads_l: int
        Number of attention heads for the larger patches
    cross_head_s: int
        Number of CrossAttention heads for the smaller patches
    cross_head_l: int
        Number of CrossAttention heads for the larger patches
    encoder_mlp_dim_s: int
        Dimension of hidden layer in the encoder for the smaller patches
    encoder_mlp_dim_l: int
        Dimension of hidden layer in the encoder for the larger patches
    in_channels: int
        Number of input channels
    decoder_config_s: int or tuple or list, optional
        Configuration of the decoder for the smaller patches
    decoder_config_l: int or tuple or list, optional
        Configuration of the decoder for the larger patches
    pool_s: str
        Feature pooling type for the smaller patches, one of {``cls``,``mean``}
    pool_l: str
        Feature pooling type for the larger patches, one of {``cls``,``mean``}
    p_dropout_encoder_s: float
        Dropout probability in the encoder for the smaller patches
    p_dropout_encoder_l: float
        Dropout probability in the encoder for the larger patches
    p_dropout_embedding_s: float
        Dropout probability in the embedding layer for the smaller patches
    p_dropout_embedding_l: float
        Dropout probability in the embedding layer for the larger patches
    """

    def __init__(self, img_size, patch_size_s, patch_size_l, n_classes, cross_dim_head_s=64, cross_dim_head_l=64, latent_dim_s=1024, latent_dim_l=1024, head_dim_s=64, head_dim_l=64, depth_s=6, depth_l=6, attn_heads_s=16, attn_heads_l=16, cross_head_s=8, cross_head_l=8, encoder_mlp_dim_s=2048, encoder_mlp_dim_l=2048, in_channels=3, decoder_config_s=None, decoder_config_l=None, pool_s='cls', pool_l='cls', p_dropout_encoder_s=0.0, p_dropout_encoder_l=0.0, p_dropout_embedding_s=0.0, p_dropout_embedding_l=0.0):
        super().__init__(img_size, patch_size_s, in_channels, pool_s)
        super().__init__(img_size, patch_size_l, in_channels, pool_l)
        self.s = _cross_p(img_size, patch_size_s, latent_dim_s, in_channels, p_dropout_embedding_s)
        self.l = _cross_p(img_size, patch_size_l, latent_dim_l, in_channels, p_dropout_embedding_l)
        self.encoder = CrossEncoder(latent_dim_s, latent_dim_l, head_dim_s, head_dim_l, cross_dim_head_s, cross_dim_head_l, depth_s, depth_l, attn_heads_s, attn_heads_l, cross_head_s, cross_head_l, encoder_mlp_dim_s, encoder_mlp_dim_l, p_dropout_encoder_s, p_dropout_encoder_l)
        self.pool_s = lambda x: x.mean(dim=1) if pool_s == 'mean' else x[:, 0]
        self.pool_l = lambda x: x.mean(dim=1) if pool_l == 'mean' else x[:, 0]
        if decoder_config_s is not None:
            if not isinstance(decoder_config_s, list):
                decoder_config_s = list(decoder_config_s)
            assert decoder_config_s[0] == latent_dim_s, '`latent_dim` should be equal to the first item of `decoder_config`'
            self.decoder_s = MLPDecoder(decoder_config_s, n_classes)
        else:
            self.decoder_s = MLPDecoder(latent_dim_s, n_classes)
        if decoder_config_l is not None:
            if not isinstance(decoder_config_l, list):
                decoder_config_l = list(decoder_config_l)
            assert decoder_config_l[0] == latent_dim_l, '`latent_dim` should be equal to the first item of `decoder_config`'
            self.decoder_l = MLPDecoder(decoder_config_l, n_classes)
        else:
            self.decoder_l = MLPDecoder(latent_dim_l, n_classes)

    def forward(self, img):
        """

        Parameters
        ----------
        img: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        emb_s = self.s(img)
        emb_l = self.l(img)
        emb_s, emb_l = self.encoder(emb_s, emb_l)
        cls_s = self.pool_s(emb_s)
        cls_l = self.pool_l(emb_l)
        n_s = self.decoder_s(cls_s)
        n_l = self.decoder_l(cls_l)
        n = n_s + n_l
        return n


class CVT(BaseClassificationModel):
    """
    Implementation of `Escaping the Big Data Paradigm with Compact Transformers <https://arxiv.org/abs/2104.05704>`_

    Parameters
    -----------
    img_size: int
        Size of the image, default is 224
    patch_size:int
        Size of the single patch in the image, default is 4
    in_channels:int
        Number of input channels in image, default is 3
    seq_pool:bool
        Whether to use sequence pooling, default is True
    embedding_dim: int
        Patch embedding dimension, default is 768
    num_layers: int
        Number of Encoders in encoder block, default is 1
    num_heads: int
        Number of heads in each transformer layer, default is 1
    mlp_ratio:float
        Ratio of mlp heads to embedding dimension, default is 4.0
    n_classes: int
        Number of classes for classification, default is 1000
    p_dropout: float
        Dropout probability, default is 0.0
    attn_dropout: float
        Dropout probability, defualt is 0.0
    drop_path: float
        Stochastic depth rate, default is 0.1
    positional_embedding: str
        One of the string values {``'learnable'``, ``'sine'`` , ``None``}, default is ``'learnable'``
    decoder_config: tuple(int) or int
        Configuration of the decoder. If None, the default configuration is used.
    """

    def __init__(self, img_size=224, patch_size=4, in_channels=3, seq_pool=True, embedding_dim=768, head_dim=96, num_layers=1, num_heads=1, mlp_ratio=4.0, n_classes=1000, p_dropout=0.1, attn_dropout=0.1, drop_path=0.1, positional_embedding='learnable', decoder_config=(768, 1024)):
        super().__init__(img_size=img_size, patch_size=patch_size)
        assert img_size % patch_size == 0, f'Image size ({img_size}) has to be divisible by patch size ({patch_size})'
        img_size = pair(img_size)
        self.in_channels = in_channels
        self.embedding = CVTEmbedding(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0, max_pool=False, activation=None, num_conv_layers=1, conv_bias=True)
        positional_embedding = positional_embedding if positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = self.embedding.sequence_length(n_channels=in_channels, height=img_size[0], width=img_size[1])
        self.seq_pool = seq_pool
        assert self.sequence_length is not None or positional_embedding == 'none', f'Positional embedding is set to {positional_embedding} and the sequence length was not specified.'
        if not seq_pool:
            self.sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
        if positional_embedding != 'none':
            self.positional_emb = PosEmbedding(shape=self.sequence_length, dim=embedding_dim, drop=p_dropout, sinusoidal=True if positional_embedding is 'sine' else False)
        else:
            self.positional_emb = None
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.encoder_blocks = nn.ModuleList([VanillaEncoder(embedding_dim=embedding_dim, num_heads=num_heads, depth=1, mlp_dim=hidden_dim, head_dim=head_dim, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=dpr[i]) for i in range(num_layers)])
        if decoder_config is not None:
            if not isinstance(decoder_config, list) and not isinstance(decoder_config, tuple):
                decoder_config = [decoder_config]
            assert decoder_config[0] == embedding_dim, f'Configurations do not match for MLPDecoder, First element of `decoder_config` expected to be {embedding_dim}, got {decoder_config[0]} '
            self.decoder = MLPDecoder(config=decoder_config, n_classes=n_classes)
        else:
            self.decoder = MLPDecoder(config=embedding_dim, n_classes=n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        x = self.embedding(x)
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.in_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if self.positional_emb is not None:
            x = self.positional_emb(x)
        for blk in self.encoder_blocks:
            x = blk(x)
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        x = self.decoder(x)
        return x


class PerceiverIO(nn.Module):
    """
    Implementation of `Perceiver IO: A General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`_

    Code Implementation based on:
    https://github.com/lucidrains/perceiver-pytorch

    Parameters
    -----------
    dim: int
        Size of sequence to be encoded
    depth: int
        Depth of latent attention blocks
    latent_dim: int
        Dimension of latent array
    num_latents: int
        Number of latent arrays
    num_cross_heads: int
        Number of heads for cross attention
    num_latent_heads: int
        Number of heads for latent attention
    cross_head_dim: int
        Dimension of cross attention head
    latent_head_dim: int
        Dimension of latent attention head
    queries_dim: int
        Dimension of queries array
    logits_dim: int, optional
        Dimension of output logits
    decoder_ff: bool
        Whether to include a feed forward layer for the decoder attention block
    """

    def __init__(self, dim=32, depth=6, latent_dim=512, num_latents=512, num_cross_heads=1, num_latent_heads=8, cross_head_dim=64, latent_head_dim=64, queries_dim=32, logits_dim=None, decoder_ff=False):
        super().__init__()
        self.encoder = PerceiverIOEncoder(dim=dim, depth=depth, latent_dim=latent_dim, num_latents=num_latents, num_cross_heads=num_cross_heads, num_latent_heads=num_latent_heads, cross_head_dim=cross_head_dim, latent_head_dim=latent_head_dim)
        self.decoder = PerceiverIODecoder(dim=dim, latent_dim=latent_dim, queries_dim=queries_dim, num_cross_heads=num_cross_heads, cross_head_dim=cross_head_dim, logits_dim=logits_dim, decoder_ff=decoder_ff)

    def forward(self, x, queries):
        out = self.encoder(x)
        out = self.decoder(out, queries=queries)
        return out


class PVTClassification(nn.Module):
    """
    Implementation of `Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolution <https://arxiv.org/abs/2102.12122>`_

    Parameters
    -----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embed_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    norm_layer:
        Normalization layer, default is nn.LayerNorm
    sr_ratios: float
        Spatial reduction ratio
    decoder_config:int or tuple[int], optional
        Configuration of the decoder. If None, the default configuration is used.
    linear: bool
        Whether to use linear Spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is False
    ape: bool
        Whether to use absolute position embedding, default is True
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, n_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], decoder_config=None, linear=False, use_dwconv=False, ape=True):
        super(PVTClassification, self).__init__()
        self.ape = ape
        self.depths = depths
        assert len(depths) == len(num_heads) == len(embed_dims), 'Configurations do not match'
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.norms = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(nn.ModuleList([OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // 2 ** (i + 1), patch_size=patch_size[i], stride=4 if i == 0 else 2, in_channels=in_channels if i == 0 else embed_dims[i - 1], embedding_dim=embed_dims[i])]))
            if ape:
                if i != len(depths) - 1:
                    self.pos_embeds.append(nn.ModuleList([PVTPosEmbedding(pos_shape=img_size // np.prod(patch_size[:i + 1]), pos_dim=embed_dims[i])]))
                else:
                    self.last_pos = nn.Parameter(torch.randn(1, (img_size // np.prod(patch_size[:i + 1])) ** 2, embed_dims[-1]))
            self.blocks.append(nn.ModuleList([PVTEncoder(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, depth=depths[i], attn_dropout=attn_dropout, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], sr_ratio=sr_ratios[i], linear=linear, activation=nn.GELU, use_dwconv=use_dwconv)]))
            self.norms.append(norm_layer(embed_dims[i]))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if decoder_config is not None:
            if not isinstance(decoder_config, list) and not isinstance(decoder_config, tuple):
                decoder_config = [decoder_config]
            assert decoder_config[0] == embed_dims[-1], f'Configurations do not match for MLPDecoder, First element of `decoder_config` expected to be {embed_dims[-1]}, got {decoder_config[0]} '
            self.decoder = MLPDecoder(config=decoder_config, n_classes=n_classes)
        else:
            self.decoder = MLPDecoder(config=embed_dims[-1], n_classes=n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        B = x.shape[0]
        for i in range(len(self.depths)):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed[0](x)
            N = x.shape[1]
            if self.ape:
                if i == len(self.depths) - 1:
                    x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
                    x += self.last_pos[:, :N + 1]
                else:
                    pos_embed = self.pos_embeds[i]
                    x = pos_embed[0](x, H=H, W=W)
            for blk in block:
                x = blk(x, H=H, W=W)
            x = norm(x)
            if i == len(self.depths) - 1:
                x = x.mean(dim=1)
            else:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.decoder(x)
        return x


class PVTClassificationV2(PVTClassification):
    """
    Implementation of `PVT v2: Improved Baselines with Pyramid Vision Transformer <https://arxiv.org/abs/2106.13797>`_

    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default is 3
    n_classes: int
        Number of classes for classification
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    norm_layer:nn.Module
        Normalization layer, default is nn.LayerNorm
    sr_ratios: float
        Spatial reduction ratio
    decoder_config:int or tuple[int], optional
        Configuration of the decoder. If None, the default configuration is used.
    linear: bool
        Whether to use linear Spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is True
    ape: bool
        Whether to use absolute position embedding, default is false
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, n_classes=1000, embedding_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=0.0, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], decoder_config=None, use_dwconv=True, linear=False, ape=False):
        super(PVTClassificationV2, self).__init__(img_size=img_size, patch_size=patch_size, in_channels=in_channels, n_classes=n_classes, embed_dims=embedding_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path_rate, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, decoder_config=decoder_config, ape=ape, use_dwconv=use_dwconv, linear=linear)


class SwinTransformer(BaseClassificationModel):
    """
    Implementation of `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` <https://arxiv.org/abs/2103.14030v1>`_

    Parameters
    -----------
    img_size: int
        Size of an Image
    patch_size: int
        Patch Size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embedding_dim: int
        Patch Embedding dimension
    depths: tuple[int]
        Depth in each Transformer layer
    num_heads: tuple[int]
        Number of heads in each transformer layer
    window_size: int
        Window Size
    mlp_ratio : float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale:  float, optional
        Override default qk scale of head_dim ** -0.5 in Window Attention if set
    p_dropout: float
        Dropout rate, default is 0.0
    attn_dropout: float
        Attention dropout rate,default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    norm_layer: nn.Module
        Normalization layer,default is nn.LayerNorm
    ape: bool, optional
        Whether to add relative/absolute position embedding to patch embedding, default is True
    decoder_config: int or tuple[int], optional
        Configuration of the decoder. If None, the default configuration is used.
    patch_norm: bool, optional
        Whether to add Normalization layer in PatchEmbedding, default is True
    """

    def __init__(self, img_size, patch_size, in_channels, n_classes, embedding_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=True, decoder_config=None, patch_norm=True):
        super(SwinTransformer, self).__init__(img_size, patch_size, in_channels, pool='mean')
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embedding_dim=embedding_dim, norm_layer=norm_layer if patch_norm else nn.Identity)
        self.patch_resolution = self.patch_embed.patch_resolution
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.ape = ape
        num_features = int(embedding_dim * 2 ** (len(depths) - 1))
        self.absolute_pos_embed = PosEmbedding(shape=num_patches, dim=embedding_dim, drop=p_dropout, std=0.02) if ape else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = SwinEncoder(dim=int(embedding_dim * 2 ** i_layer), input_resolution=(self.patch_resolution[0] // 2 ** i_layer, self.patch_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qkv_scale=qk_scale, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < len(depths) - 1 else None)
            self.encoder.append(layer)
        if decoder_config is not None:
            if not isinstance(decoder_config, list):
                decoder_config = list(decoder_config)
            assert decoder_config[0] == num_features, f'first item of `decoder_config` should be equal to the `num_features`; num_features=embed_dim * 2** (len(depths)-1) which is = {num_features} '
            self.decoder = MLPDecoder(decoder_config, n_classes)
        else:
            self.decoder = MLPDecoder(num_features, n_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm_layer(num_features) if norm_layer is not None else nn.Identity
        self.pos_drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        x = self.patch_embed(x)
        x = self.absolute_pos_embed(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).flatten(1)
        x = self.decoder(x)
        return x


class VanillaViT(BaseClassificationModel):
    """
    Implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Parameters
    -----------
    img_size: int
        Size of the image
    patch_size: int
        Size of a patch
    n_classes: int
        Number of classes for classification
    embedding_dim: int
        Dimension of hidden layer
    head_dim: int
        Dimension of the attention head
    depth: int
        Number of attention layers in the encoder
    num_heads:int
        Number of the attention heads
    encoder_mlp_dim: int
        Dimension of hidden layer in the encoder
    in_channels: int
        Number of input channels
    decoder_config: int or tuple or list, optional
        Configuration of the decoder. If None, the default configuration is used.
    pool: str
        Feature pooling type, one of {``cls``,``mean``}
    p_dropout_encoder: float
        Dropout probability in the encoder
    p_dropout_embedding: float
        Dropout probability in the embedding layer
    """

    def __init__(self, img_size, patch_size, n_classes, embedding_dim=1024, head_dim=64, depth=6, num_heads=16, encoder_mlp_dim=2048, in_channels=3, decoder_config=None, pool='cls', p_dropout_encoder=0.0, p_dropout_embedding=0.0):
        super().__init__(img_size, patch_size, in_channels, pool)
        self.patch_embedding = LinearEmbedding(embedding_dim, self.patch_height, self.patch_width, self.patch_dim)
        self.pos_embedding = PosEmbedding(shape=self.num_patches + 1, dim=embedding_dim, drop=p_dropout_embedding, sinusoidal=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.encoder = VanillaEncoder(embedding_dim=embedding_dim, depth=depth, num_heads=num_heads, head_dim=head_dim, mlp_dim=encoder_mlp_dim, p_dropout=p_dropout_encoder)
        self.pool = lambda x: x.mean(dim=1) if pool == 'mean' else x[:, 0]
        if decoder_config is not None:
            if not isinstance(decoder_config, list):
                decoder_config = list(decoder_config)
            assert decoder_config[0] == embedding_dim, '`embedding_dim` should be equal to the first item of `decoder_config`'
            self.decoder = MLPDecoder(decoder_config, n_classes)
        else:
            self.decoder = MLPDecoder(embedding_dim, n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.decoder(x)
        return x


class VisformerConvBlock(nn.Module):
    """
    Convolution Block for Vision-Friendly transformers
    https://arxiv.org/abs/2104.12533

    Parameters
    -----------
    in_channels: int
        Number of input channels
    group: int
        Number of groups for convolution, default is 8
    activation: torch.nn.Module
        Activation function between layers, default is nn.GELU
    p_dropout: float
        Dropout rate, default is 0.0
    """

    def __init__(self, in_channels, group=8, activation=nn.GELU, p_dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, groups=group, bias=False)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of same size as input
        """
        xt = x
        xt = self.norm1(xt)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x + xt
        return x


class VisformerAttentionBlock(nn.Module):
    """
    Attention Block for Vision-Friendly transformers
    https://arxiv.org/abs/2104.12533

    Parameters
    ----------
    in_channels: int
        Number of input channels
    num_heads: int
        Number of heads for attention, default is 8
    activation: torch.nn.Module
        Activation function between layers, default is nn.GELU
    p_dropout: float
        Dropout rate, default is 0.0
    """

    def __init__(self, in_channels, num_heads=8, activation=nn.GELU, p_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False)
        self.attn = VanillaSelfAttention(in_channels, num_heads=num_heads, head_dim=in_channels // num_heads)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of same size as input
        """
        B, C, H, W = x.shape
        xt = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.attn(x)
        x = xt + x
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        xt = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = xt + x
        return x


class Visformer(nn.Module):
    """
    A builder to construct a Vision-Friendly transformer model as in the paper :`Visformer: The Vision-friendly Transformer <https://arxiv.org/abs/1906.11488>`_

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    depth: tuple[int]
        Number of layers before each embedding reduction
    config: tuple[int]
        Choice of convolution block (0) or attention block (1) for corresponding layer
    channel_config: tuple[int]
        Number of channels for each layer
    num_heads: int
        Number of heads for attention block, default is 8
    conv_group: int
        Number of groups for convolution block, default is 8
    p_dropout_conv: float
        Dropout rate for convolution block, default is 0.0
    p_dropout_attn: float
        Dropout rate for attention block, default is 0.0
    activation: nn.Module
        Activation function between layers, default is nn.GELU
    pos_embedding: bool
        Whether to use positional embedding, default is True

    """

    def __init__(self, img_size, n_classes, depth: tuple, config: tuple, channel_config: tuple, num_heads=8, conv_group=8, p_dropout_conv=0.0, p_dropout_attn=0.0, activation=nn.GELU, pos_embedding=True):
        super().__init__()
        q = 0
        assert len(channel_config) == len(depth) - depth.count(0) + 2, 'Channel config is not correct'
        assert set(config).issubset(set([0, 1])), 'Config is not correct, should contain only 0 and 1'
        self.linear = nn.Linear(channel_config[-1], n_classes)
        if isinstance(img_size, int):
            img_size = img_size, img_size
        image_size = list(img_size)
        assert image_size[0] // 2 ** (len(depth) + 1) > 0, 'Image size is too small'
        assert image_size[1] // 2 ** (len(depth) + 1) > 0, 'Image size is too small'
        self.stem = nn.ModuleList([nn.Conv2d(channel_config[q], channel_config[q + 1], kernel_size=7, padding=3, stride=2, bias=False), nn.BatchNorm2d(channel_config[q + 1]), nn.ReLU(inplace=True)])
        q += 1
        emb = 2
        image_size = [(i // 2) for i in image_size]
        for i in range(len(depth)):
            if depth[i] == 0:
                emb *= 2
                config = tuple([0] + list(config))
                continue
            self.stem.extend([nn.Conv2d(channel_config[q], channel_config[q + 1], kernel_size=emb, stride=emb), nn.BatchNorm2d(channel_config[q + 1]), nn.ReLU(inplace=True)])
            image_size = [(k // emb) for k in image_size]
            emb = 2
            q += 1
            if pos_embedding:
                self.stem.extend([PosEmbedding([channel_config[q], image_size[0]], image_size[1])])
            if config[i] == 0:
                self.stem.extend([VisformerConvBlock(channel_config[q], group=conv_group, p_dropout=p_dropout_conv, activation=activation) for j in range(depth[i])])
            elif config[i] == 1:
                self.stem.extend([VisformerAttentionBlock(channel_config[q], num_heads, activation, p_dropout_attn) for j in range(depth[i])])
        self.stem.extend([nn.BatchNorm2d(channel_config[-1]), nn.AdaptiveAvgPool2d(1)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        for i in self.stem:
            x = i(x)
        x.squeeze_(2).squeeze_(2)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class ViViTModel2(BaseClassificationModel):
    """
    Model 2 implementation of: `ViViT: A Video Vision Transformer <https://arxiv.org/abs/2103.15691>`_

    Parameters
    -----------
    img_size:int
        Size of single frame/ image in video
    in_channels:int
        Number of channels
    patch_size: int
        Patch size
    embedding_dim: int
        Embedding dimension of a patch
    num_frames:int
        Number of seconds in each Video
    depth:int
        Number of encoder layers
    num_heads:int
        Number of attention heads
    head_dim:int
        Dimension of head
    n_classes:int
        Number of classes
    mlp_dim: int
        Dimension of hidden layer
    pool: str
        Pooling operation,must be one of {"cls","mean"},default is "cls"
    p_dropout:float
        Dropout probability
    attn_dropout:float
        Dropout probability
    drop_path_rate:float
        Stochastic drop path rate
    """

    def __init__(self, img_size, in_channels, patch_size, embedding_dim, num_frames, depth, num_heads, head_dim, n_classes, mlp_dim=None, pool='cls', p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.02):
        super(ViViTModel2, self).__init__(img_size=img_size, in_channels=in_channels, patch_size=patch_size, pool=pool)
        patch_dim = in_channels * patch_size ** 2
        self.patch_embedding = LinearVideoEmbedding(embedding_dim=embedding_dim, patch_height=patch_size, patch_width=patch_size, patch_dim=patch_dim)
        self.pos_embedding = PosEmbedding(shape=[num_frames, self.num_patches + 1], dim=embedding_dim, drop=p_dropout)
        self.space_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.spatial_transformer = VanillaEncoder(embedding_dim=embedding_dim, depth=depth, num_heads=num_heads, head_dim=head_dim, mlp_dim=mlp_dim, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path_rate)
        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.temporal_transformer = VanillaEncoder(embedding_dim=embedding_dim, depth=depth, num_heads=num_heads, head_dim=head_dim, mlp_dim=mlp_dim, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path_rate)
        self.decoder = MLPDecoder(config=[embedding_dim], n_classes=n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, t, n, d = x.shape
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = nn.Parameter(torch.cat((cls_space_tokens, x), dim=2))
        x = self.pos_embedding(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        cls_temporal_tokens = repeat(self.time_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.decoder(x)
        return x


class ViViTModel3(BaseClassificationModel):
    """
    Model 3 Implementation from : `ViViT: A Video Vision Transformer <https://arxiv.org/abs/2103.15691>`_

    Parameters
    ----------
    img_size:int or tuple[int]
        size of a frame
    patch_t:int
        Temporal length of single tube/patch in tubelet embedding
    patch_h:int
        Height  of single tube/patch in tubelet embedding
    patch_w:int
        Width  of single tube/patch in tubelet embedding
    in_channels: int
        Number of input channels, default is 3
    n_classes:int
        Number of classes
    num_frames :int
        Number of seconds in each Video
    embedding_dim:int
        Embedding dimension of a patch
    depth:int
        Number of Encoder layers
    num_heads: int
        Number of attention heads
    head_dim:int
        Dimension of attention head
    p_dropout:float
        Dropout rate/probability, default is 0.0
    mlp_dim: int
        Hidden dimension, optional
    """

    def __init__(self, img_size, patch_t, patch_h, patch_w, in_channels, n_classes, num_frames, embedding_dim, depth, num_heads, head_dim, p_dropout, mlp_dim=None):
        super(ViViTModel3, self).__init__(in_channels=in_channels, patch_size=(patch_h, patch_w), pool='mean', img_size=img_size)
        h, w = pair(img_size)
        self.tubelet_embedding = TubeletEmbedding(embedding_dim=embedding_dim, tubelet_t=patch_t, tubelet_h=patch_h, tubelet_w=patch_w, in_channels=in_channels)
        self.pos_embbedding = PosEmbedding(shape=[num_frames // patch_t, h * w // (patch_w * patch_h)], dim=embedding_dim)
        self.encoder = ViViTEncoder(dim=embedding_dim, num_heads=num_heads, head_dim=head_dim, p_dropout=p_dropout, depth=depth, hidden_dim=mlp_dim)
        self.decoder = MLPDecoder(config=[embedding_dim], n_classes=n_classes)

    def forward(self, x):
        x = self.tubelet_embedding(x)
        x = self.pos_embbedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x


class PVTDetection(nn.Module):
    """
    Implementation of Pyramid Vision Transformer:
    https://arxiv.org/abs/2102.12122v1


    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding, default is False
    ape: bool
        Whether to use absolute position embedding, default is True

    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, embedding_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=False, use_dwconv=False, ape=True):
        super(PVTDetection, self).__init__()
        self.ape = ape
        self.depths = depths
        assert len(depths) == len(num_heads) == len(embedding_dims), 'Configurations do not match'
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.norms = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(nn.ModuleList([OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // 2 ** (i + 1), patch_size=patch_size[i], stride=4 if i == 0 else 2, in_channels=in_channels if i == 0 else embedding_dims[i - 1], embedding_dim=embedding_dims[i])]))
            if ape:
                self.pos_embeds.append(nn.ModuleList([PVTPosEmbedding(pos_shape=img_size // np.prod(patch_size[:i + 1]), pos_dim=embedding_dims[i])]))
            self.blocks.append(nn.ModuleList([PVTEncoder(dim=embedding_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, depth=depths[i], attn_dropout=attn_dropout, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], sr_ratio=sr_ratios[i], linear=linear, activation=nn.GELU, use_dwconv=use_dwconv)]))
            self.norms.append(norm_layer(embedding_dims[i]))
        self.pool = nn.Parameter(torch.zeros(1, 1, embedding_dims[-1]))

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns list containing output features from all pyramid stages

        """
        B = x.shape[0]
        out = []
        for i in range(len(self.depths)):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed[0](x)
            if self.ape:
                pos_embed = self.pos_embeds[i]
                x = pos_embed[0](x, H=H, W=W)
            for blk in block:
                x = blk(x, H=H, W=W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out.append(x)
        return out


class PVTDetectionV2(PVTDetection):
    """
    Implementation of Pyramid Vision Transformer:
    https://arxiv.org/abs/2102.12122v2


    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding
    ape: bool
        Whether to use absolute position embedding
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, embedding_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=0.0, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], ape=False, use_dwconv=True, linear=False):
        super(PVTDetectionV2, self).__init__(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embedding_dims=embedding_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path_rate, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, linear=linear, ape=ape, use_dwconv=use_dwconv)


class PVTSegmentation(nn.Module):
    """
    Using Pyramid Vision Transformer as a backbone for a segmentation model with help of U-Net like segmentation head.
    https://arxiv.org/abs/2102.12122v1

    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float
        Dropout rate,default is 0.0
    attn_dropout:  float
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding
    ape: bool
        Whether to use absolute position embedding
    return_pyramid:bool
        Whether to use all pyramid feature layers for up-sampling, default is False
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, embedding_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=False, out_channels=1, use_dwconv=False, ape=True, return_pyramid=False):
        super(PVTSegmentation, self).__init__()
        self.ape = ape
        self.depths = depths
        self.return_pyramid = return_pyramid
        assert len(depths) == len(num_heads) == len(embedding_dims), 'Configurations do not match'
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.norms = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(nn.ModuleList([OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // 2 ** (i + 1), patch_size=patch_size[i], stride=4 if i == 0 else 2, in_channels=in_channels if i == 0 else embedding_dims[i - 1], embedding_dim=embedding_dims[i])]))
            if ape:
                self.pos_embeds.append(nn.ModuleList([PVTPosEmbedding(pos_shape=img_size // np.prod(patch_size[:i + 1]), pos_dim=embedding_dims[i])]))
            self.blocks.append(nn.ModuleList([PVTEncoder(dim=embedding_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, depth=depths[i], attn_dropout=attn_dropout, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], sr_ratio=sr_ratios[i], linear=linear, activation=nn.GELU, use_dwconv=use_dwconv)]))
            self.norms.append(norm_layer(embedding_dims[i]))
        self.head = SegmentationHead(out_channels=out_channels, embed_dims=embedding_dims if not return_pyramid else [embedding_dims[-1]])

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor
        """
        B = x.shape[0]
        out = []
        for i in range(len(self.depths)):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed[0](x)
            if self.ape:
                pos_embed = self.pos_embeds[i]
                x = pos_embed[0](x, H=H, W=W)
            for blk in block:
                x = blk(x, H=H, W=W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out.append(x)
        if self.return_pyramid:
            out = out[3:4]
        out = self.head(out)
        return out


class PVTSegmentationV2(PVTSegmentation):
    """
    Using Pyramid Vision Transformer as a backbone for a segmentation model with help of U-Net like segmentation head.

    https://arxiv.org/abs/2106.13797


    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding, default is True
    ape: bool
        Whether to use absolute position embedding, default is False
    return_pyramid: bool
        Whether to use all pyramid feature layers for up-sampling, default is true
    """

    def __init__(self, img_size=224, patch_size=[7, 3, 3, 3], in_channels=3, embedding_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[4, 4, 4, 4], qkv_bias=False, qk_scale=0.0, p_dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], ape=False, use_dwconv=True, linear=False, return_pyramid=False):
        super(PVTSegmentationV2, self).__init__(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embedding_dims=embedding_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, p_dropout=p_dropout, attn_dropout=attn_dropout, drop_path_rate=drop_path_rate, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, linear=linear, ape=ape, use_dwconv=use_dwconv, return_pyramid=return_pyramid)


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module

    Parameters
    -----------
    features :int
        Number of features
    activation: nn.Module
        Activation module, default is nn.GELU
    bn: bool
        Whether to use batch normalisation
    """

    def __init__(self, features, activation=nn.GELU, bn=True):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """forward pass"""
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass"""
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class Interpolate(nn.Module):
    """Interpolation module

    Parameters
    -----------
    scale_factor : float
        Scaling factor used in interpolation
    mode :str
        Interpolation mode
    align_corners: bool
        Whether to align corners in Interpolation operation
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass"""
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = posemb[:, :self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


activations = {}


attention = {}


def forward_flex(self, x):
    b, c, h, w = x.shape
    pos_embed = self.model._resize_pos_embed(self.model.pos_embedding.pos_embed, h // self.model.patch_size[1], w // self.model.patch_size[0])
    B = x.shape[0]
    x = self.model.patch_embedding.patch_embedding(x)
    cls_tokens = self.model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.model.pos_embedding.pos_drop(x)
    x = self.model.encoder(x)
    return x


def get_activation(name):

    def hook(model, input, output):
        activations[name] = output
    return hook


def get_attention(name):

    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.to_qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * module.scale
        attn = attn.softmax(dim=-1)
        attention[name] = attn
    return hook


class AddReadout(nn.Module):
    """Handles readout operation when `readout` parameter is `add`. Removes `cls_token` or  `readout_token` from tensor and adds it to the rest of tensor"""

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    """Another class that handles readout operation. Used when `readout` parameter is `project`"""

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Slice(nn.Module):
    """Handles readout operation when `readout` parameter is `ignore`. Removes `cls_token` or  `readout_token` by index slicing"""

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    else:
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
    return readout_oper


class DPTDepth(nn.Module):
    """
    Implementation of " Vision Transformers for Dense Prediction "
    https://arxiv.org/abs/2103.13413

    Parameters
    -----------
    backbone:str
        Name of ViT model to be used as backbone, must be one of {`vitb16`,`vitl16`,`vit_tiny`}
    in_channels: int
        Number of channels in input image, default is 3
    img_size: tuple[int]
        Input image size, default is (384,384)
    readout:str
        Method to handle the `readout_token` or `cls_token`
        Must be one of {`add`, `ignore`,`project`}, default is `project`
    hooks: list[int]
        List representing index of encoder blocks on which hooks will be registered.
        These hooks extract features from different ViT blocks, eg attention, default is (2,5,8,11).
    channels_last: bool
        Alters the memory format of storing tensors, default is False,
        For more information visit, this `blogpost<https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`
    use_bn:bool
        If True, BatchNormalisation is used in `FeatureFusionBlock_custom`, default is False
    enable_attention_hooks:bool
        If True, `get_attention` hook is registered, default is false
    non_negative:bool
        If True, Relu operation will be applied in `DPTDepth.model.head` block, default is True
    invert:bool
        If True, forward pass output of `DPTDepth.model.head` will be transformed (inverted)
        according to `scale` and `shift` parameters, default is False
    scale:float
        Float value that will be multiplied with forward pass output from `DPTDepth.model.head`, default is 1.0
    shift:float
        Float value that will be added with forward pass output from `DPTDepth.model.head` after scaling, default is 0.0
    """

    def __init__(self, backbone, in_channels=3, img_size=(384, 384), readout='project', hooks=(2, 5, 8, 11), channels_last=False, use_bn=False, enable_attention_hooks=False, non_negative=True, scale=1.0, shift=0.0, invert=False):
        super(DPTDepth, self).__init__()
        self.channels_last = channels_last
        self.use_bn = use_bn
        self.enable_attention_hooks = enable_attention_hooks
        self.non_negative = non_negative
        self.scale = scale
        self.shift = shift
        self.invert = invert
        start_index = 1
        if backbone == 'vitb16':
            scratch_in_features = 96, 192, 384, 768
            self.model = MODEL_REGISTRY.get('VanillaViT')(img_size=img_size, patch_size=16, embedding_dim=768, head_dim=64, depth=12, num_heads=12, encoder_mlp_dim=768, n_classes=10, in_channels=in_channels)
            hooks = (2, 5, 8, 11) if hooks is None else hooks
            self.vit_features = 768
        elif backbone == 'vitl16':
            scratch_in_features = 256, 512, 1024, 1024
            self.model = MODEL_REGISTRY.get('VanillaViT')(img_size=img_size, patch_size=16, embedding_dim=1024, head_dim=64, depth=24, num_heads=16, encoder_mlp_dim=1024, n_classes=10, in_channels=in_channels)
            hooks = (5, 11, 17, 23) if hooks is None else hooks
            self.vit_features = 1024
        elif backbone == 'vit_tiny':
            scratch_in_features = 48, 96, 144, 192
            self.model = MODEL_REGISTRY.get('VanillaViT')(img_size=img_size, patch_size=16, embedding_dim=192, head_dim=64, depth=12, num_heads=3, encoder_mlp_dim=192, n_classes=3, in_channels=in_channels)
            hooks = (2, 5, 8, 11) if hooks is None else hooks
            self.vit_features = 192
        else:
            raise NotImplementedError
        assert readout in ('add', 'ignore', 'project'), f"Not valid `readout` param, Must be one of ('add','ignore','project'), but got {readout}"
        features = scratch_in_features[0]
        self._register_hooks_and_add_postprocess(size=img_size, features=scratch_in_features, hooks=hooks, use_readout=readout, enable_attention_hooks=enable_attention_hooks, start_index=start_index)
        self._make_scratch(in_shape=scratch_in_features, out_shape=features, groups=1, expand=False)
        self._add_refinenet_to_scratch(features=features, use_bn=use_bn)
        self.model.head = nn.Sequential(nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), Interpolate(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(True) if non_negative else nn.Identity(), nn.Identity())

    def _register_hooks_and_add_postprocess(self, size=(384, 384), features=(96, 192, 384, 768), hooks=(2, 5, 8, 11), use_readout='ignore', enable_attention_hooks=False, start_index=1):
        """
        Registers forward hooks to the backbone and initializes activation-postprocessing-blocks (act_postprocess(int))
        Parameters
        -----------
        size: tuple[int]
            Input image size
        features:tuple[int]
            Number of features
        hooks:tuple[int]
            List containing index of encoder blocks to which forward hooks will be registered
        use_readout:str
            Appropriate readout operation,must be one of  {`add`,`ignore`,`project`}
        enable_attention_hooks:bool
            If True, forward hooks will be registered to attention blocks.
        start_index:int
            Parameter that handles readout operation, default value is 1.
        """
        for i in range(4):
            self.model.encoder.encoder[hooks[i]][0].fn.register_forward_hook(get_activation(str(i + 1)))
        self.activations = activations
        if enable_attention_hooks:
            for i in range(4):
                self.model.encoder.encoder[hooks[i]][0].fn.register_forward_hook(get_attention(f'attn_{str(i + 1)}'))
            self.attention = attention
        readout_oper = get_readout_oper(self.vit_features, features, use_readout, start_index)
        self.act_postprocess1 = nn.Sequential(readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=self.vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
        self.act_postprocess2 = nn.Sequential(readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=self.vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
        self.act_postprocess3 = nn.Sequential(readout_oper[2], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=self.vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0))
        self.act_postprocess4 = nn.Sequential(readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=self.vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
        self.model.start_index = start_index
        self.model.patch_size = [16, 16]
        self.model.forward_flex = types.MethodType(forward_flex, self.model)
        self.model._resize_pos_embed = types.MethodType(_resize_pos_embed, self.model)

    def _make_scratch(self, in_shape, out_shape, groups=1, expand=False):
        """
        Makes a scratch module which is subclass of nn.Module

        Parameters
        -----------
        in_shape: list[int]
        out_shape:int
        groups: int
        expand:bool
        """
        self.scratch = nn.Module()
        for i in range(4):
            layer = nn.Conv2d(in_shape[i], out_shape * 2 ** i if expand else out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
            setattr(self.scratch, f'layer{i + 1}_rn', layer)

    def _add_refinenet_to_scratch(self, features, use_bn):
        """

        Parameters
        -----------
        features: int
            Number of features
        use_bn: bool
            Whether to use batch normalisation
        """
        for i in range(4):
            refinenet = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
            setattr(self.scratch, f'refinenet{i + 1}', refinenet)

    def forward_vit(self, x):
        """
        Performs forward pass on backbone ViT model and fetches output from different encoder blocks with the help of hooks

        Parameters
        -----------
        x: torch.Tensor
            Input image tensor
        """
        b, c, h, w = x.shape
        glob = forward_flex(self, x)
        layer_1 = self.activations['1']
        layer_2 = self.activations['2']
        layer_3 = self.activations['3']
        layer_4 = self.activations['4']
        layer_1 = self.act_postprocess1[0:2](layer_1)
        layer_2 = self.act_postprocess2[0:2](layer_2)
        layer_3 = self.act_postprocess3[0:2](layer_3)
        layer_4 = self.act_postprocess4[0:2](layer_4)
        unflatten = nn.Sequential(nn.Unflatten(2, torch.Size([h // self.model.patch_size[1], w // self.model.patch_size[0]])))
        if layer_1.ndim == 3:
            layer_1 = unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = unflatten(layer_4)
        layer_1 = self.act_postprocess1[3:len(self.act_postprocess1)](layer_1)
        layer_2 = self.act_postprocess2[3:len(self.act_postprocess2)](layer_2)
        layer_3 = self.act_postprocess3[3:len(self.act_postprocess3)](layer_3)
        layer_4 = self.act_postprocess4[3:len(self.act_postprocess4)](layer_4)
        return layer_1, layer_2, layer_3, layer_4

    def forward(self, x):
        """
        Forward pass of DPTDepth

        Parameters
        -----------
        x:torch.Tensor
            Input image tensor
        """
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)
        layer_1, layer_2, layer_3, layer_4 = self.forward_vit(x)
        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)
        path1 = self.scratch.refinenet4(layer_4)
        path1 = self.scratch.refinenet3(path1, layer_3)
        path1 = self.scratch.refinenet2(path1, layer_2)
        path1 = self.scratch.refinenet1(path1, layer_1)
        inv_depth = self.model.head(path1).squeeze(dim=1)
        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-08] = 1e-08
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CVTEmbedding,
     lambda: ([], {'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OverlapPatchEmbed,
     lambda: ([], {'img_size': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PVTClassification,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PVTClassificationV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PVTDetection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PVTDetectionV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PVTFeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PVTSegmentation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PVTSegmentationV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PatchEmbedding,
     lambda: ([], {'img_size': 4, 'patch_size': 4, 'in_channels': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PerceiverIODecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PosEmbedding,
     lambda: ([], {'shape': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProjectReadout,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Slice,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialAttention,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (VisformerConvBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SforAiDl_vformer(_paritybench_base):
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

