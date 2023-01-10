import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
A2Atttention = _module
ACmixAttention = _module
AFT = _module
Axial_attention = _module
BAM = _module
CBAM = _module
CoAtNet = _module
CoTAttention = _module
CoordAttention = _module
CrissCrossAttention = _module
Crossformer = _module
DANet = _module
DAT = _module
ECAAttention = _module
EMSA = _module
ExternalAttention = _module
HaloAttention = _module
MOATransformer = _module
MUSEAttention = _module
MobileViTAttention = _module
MobileViTv2Attention = _module
OutlookAttention = _module
PSA = _module
ParNetAttention = _module
PolarizedSelfAttention = _module
ResidualAttention = _module
S2Attention = _module
SEAttention = _module
SGE = _module
SKAttention = _module
SelfAttention = _module
ShuffleAttention = _module
SimAM = _module
SimplifiedSelfAttention = _module
TripletAttention = _module
UFOAttention = _module
ViP = _module
gfnet = _module
CMT = _module
CPVT = _module
CaiT = _module
CeiT = _module
CoaT = _module
ConTNet = _module
ConViT = _module
Container = _module
ConvMixer = _module
CrossViT = _module
DViT = _module
DeiT = _module
EfficientFormer = _module
HATNet = _module
LeViT = _module
MobileNetV3 = _module
MobileViT = _module
PIT = _module
PVT = _module
PatchConvnet = _module
ShuffleTransformer = _module
TnT = _module
VOLO = _module
resnet = _module
resnext = _module
swin_transformer = _module
swin_transformer_v2 = _module
swin_transformer_v2_cr = _module
CondConv = _module
DepthwiseSeparableConvolution = _module
DynamicConv = _module
HorNet = _module
Involution = _module
MBConv = _module
g_mlp = _module
mlp_mixer = _module
repmlp = _module
resmlp = _module
sMLP_block = _module
acnet = _module
ddb = _module
mobileone = _module
repvgg = _module
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


from torch import nn


from torch.nn import functional as F


import numpy as np


from torch.nn import init


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


from torch import sqrt


from math import sqrt


from torch import flatten


from torch.nn.modules.activation import ReLU


from torch.nn.modules.batchnorm import BatchNorm2d


from torch.nn import Softmax


import torch.utils.checkpoint as checkpoint


import math


from collections import OrderedDict


from torch import einsum


from torch.nn.parameter import Parameter


from torch.functional import norm


import logging


from functools import partial


from typing import Any


from typing import List


import warnings


import matplotlib.pyplot as plt


from torch.nn.modules.activation import GELU


from torch.nn.modules.pooling import AdaptiveAvgPool2d


import torch.hub


import torch.nn.init as init


import copy


from typing import Dict


import itertools


from torch.nn.modules import conv


from torch.nn.modules.conv import Conv2d


from typing import Optional


from typing import Tuple


from typing import Union


from typing import Type


import torch.fft


from torch import select


from numpy import random


from torch import mean


from torch import conv2d


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1))
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))
        tmpZ = global_descriptors.matmul(attention_vectors)
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


class ACmix(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.0
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride
        pe = self.conv_p(position(h, w, x.is_cuda))
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)
        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w), v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)
        return self.rate1 * out_att + self.rate2 * out_conv


class AFT_FULL(nn.Module):

    def __init__(self, d_model, n=49, simple=False):
        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        bs, n, dim = input.shape
        q = self.fc_q(input)
        k = self.fc_k(input).view(1, bs, n, dim)
        v = self.fc_v(input).view(1, bs, n, dim)
        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)
        out = numerator / denominator
        out = self.sigmoid(q) * out.permute(1, 0, 2)
        return out


class Deterministic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)
        return x, dx


class IrreversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)


class _ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


class ChanLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class Sequential(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


class PermuteToFrom(nn.Module):

    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):

    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]
        self.num_axials = len(shape)
        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


class SelfAttention(nn.Module):

    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = dim // heads if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1)
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bie,bje->bij', q, k) * e ** -0.5
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else emb_dim + total_dimensions
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]
    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
    return permutations


class AxialAttention(nn.Module):

    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert dim % heads == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else dim_index + self.total_dimensions
        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'
        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


def exists(val):
    return val is not None


class AxialImageTransformer(nn.Module):

    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)
        get_ff = lambda : nn.Sequential(ChanLayerNorm(dim), nn.Conv2d(dim, dim * 4, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv2d(dim * 4, dim, 3, padding=1))
        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)
        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False), nn.ReLU(), nn.Conv2d(channel // reduction, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class BAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """
    计算出 Conv2dSamePadding with a stride.
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    return partial(Conv2dStaticSamePadding, image_size=image_size)


class MBConvBlock(nn.Module):
    """
    层 ksize3*3 输入32 输出16  conv1  stride步长1
    """

    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, image_size=224):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride
        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._kernel_size
        s = self._stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class CoAtNet(nn.Module):

    def __init__(self, in_ch, image_size, out_chs=[64, 96, 192, 384, 768]):
        super().__init__()
        self.out_chs = out_chs
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.s0 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
        self.mlp0 = nn.Sequential(nn.Conv2d(in_ch, out_chs[0], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[0], out_chs[0], kernel_size=1))
        self.s1 = MBConvBlock(ksize=3, input_filters=out_chs[0], output_filters=out_chs[0], image_size=image_size // 2)
        self.mlp1 = nn.Sequential(nn.Conv2d(out_chs[0], out_chs[1], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[1], out_chs[1], kernel_size=1))
        self.s2 = MBConvBlock(ksize=3, input_filters=out_chs[1], output_filters=out_chs[1], image_size=image_size // 4)
        self.mlp2 = nn.Sequential(nn.Conv2d(out_chs[1], out_chs[2], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[2], out_chs[2], kernel_size=1))
        self.s3 = ScaledDotProductAttention(out_chs[2], out_chs[2] // 8, out_chs[2] // 8, 8)
        self.mlp3 = nn.Sequential(nn.Linear(out_chs[2], out_chs[3]), nn.ReLU(), nn.Linear(out_chs[3], out_chs[3]))
        self.s4 = ScaledDotProductAttention(out_chs[3], out_chs[3] // 8, out_chs[3] // 8, 8)
        self.mlp4 = nn.Sequential(nn.Linear(out_chs[3], out_chs[4]), nn.ReLU(), nn.Linear(out_chs[4], out_chs[4]))

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.mlp0(self.s0(x))
        y = self.maxpool2d(y)
        y = self.mlp1(self.s1(y))
        y = self.maxpool2d(y)
        y = self.mlp2(self.s2(y))
        y = self.maxpool2d(y)
        y = y.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)
        y = self.mlp3(self.s3(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mlp4(self.s4(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1))
        N = y.shape[-1]
        y = y.reshape(B, self.out_chs[4], int(sqrt(N)), int(sqrt(N)))
        return y


class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False), nn.BatchNorm2d(dim), nn.ReLU())
        self.value_embed = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
        factor = 4
        self.attention_embed = nn.Sequential(nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False), nn.BatchNorm2d(2 * dim // factor), nn.ReLU(), nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1))

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float('inf')).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


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


class DynamicPosBias(nn.Module):

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.num_heads))

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):

    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False), nn.ReLU(), nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False))
        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)
        att = self.net(att).view(x.shape[0], -1)
        return F.softmax(att / self.temprature, -1)


class CrossFormerBlock(nn.Module):
    """ CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, group_size=7, lsda_flag=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        if min(self.input_resolution) <= self.group_size:
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, group_size=to_2tuple(self.group_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size %d, %d, %d' % (L, H, W)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        G = self.group_size
        if self.lsda_flag == 0:
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
        else:
            x = x.reshape(B, G, H // G, G, W // G, C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * H * W // G ** 2, G ** 2, C)
        x = self.attn(x, mask=self.attn_mask)
        x = x.reshape(B, H // G, W // G, G, G, C)
        if self.lsda_flag == 0:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


def bhwc_to_bchw(x: torch.Tensor) ->torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


class PatchMerging(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized.
    """

    def __init__(self, dim: int, norm_layer: Type[nn.Module]=nn.LayerNorm) ->None:
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 5, 3, 1).flatten(3)
        x = self.norm(x)
        x = bhwc_to_bchw(self.reduction(x))
        return x


class Stage(nn.Module):
    """ CrossFormer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): variable G in the paper, one group has GxG embeddings
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, group_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, patch_size_end=[4], num_patch_size=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if i % 2 == 0 else 1
            self.blocks.append(CrossFormerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, group_size=group_size, lsda_flag=lsda_flag, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, num_patch_size=num_patch_size))
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


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


class CrossFormer(nn.Module):
    """ CrossFormer
        A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], group_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, merge_size=[[2], [2], [2]], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
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
        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], group_size=group_size[i_layer], mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, patch_size_end=patch_size_end, num_patch_size=num_patch_size)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
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
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class PositionAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)
        y = self.pa(y, y, y)
        return y


class SimplifiedScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(SimplifiedScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class ChannelAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)
        y = self.pa(y, y, y)
        return y


class DAModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=512, kernel_size=3, H=7, W=7)
        self.channel_attention_module = ChannelAttentionModule(d_model=512, kernel_size=3, H=7, W=7)

    def forward(self, input):
        bs, c, h, w = input.shape
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        c_out = c_out.view(bs, c, h, w)
        return p_out + c_out


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads))
        trunc_normal_(self.relative_position_bias_table, std=0.01)
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

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1])
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')
        qkv = self.proj_qkv(x_total)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)
        if mask is not None:
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))
        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1])
        return x, None, None


class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)
        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size
        assert 0 < self.shift_size < min(self.window_size), 'wrong shift size.'
        img_mask = torch.zeros(*self.fmap_size)
        h_slices = slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0], w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x, None, None


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttentionBaseline(nn.Module):

    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups, attn_drop, proj_drop, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, stage_idx):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]
        self.conv_offset = nn.Sequential(nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels), LayerNormProxy(self.n_group_channels), nn.GELU(), nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False))
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        if self.no_off:
            offset = offset.fill(0.0)
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        x_sampled = F.grid_sample(input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1), grid=displacement[..., (1, 0)], mode='bilinear', align_corners=True)
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        y = self.proj_drop(self.proj_out(out))
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        return x


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt, dim_in, dim_embed, depths, stage_spec, n_groups, use_pe, sr_ratio, heads, stride, offset_range_factor, stage_idx, dwc_pe, no_off, fixed_pe, attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):
        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.layer_norms = nn.ModuleList([LayerNormProxy(dim_embed) for _ in range(2 * depths)])
        self.mlps = nn.ModuleList([(TransformerMLPWithConv(dim_embed, expansion, drop) if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)) for _ in range(depths)])
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop))
            elif stage_spec[i] == 'D':
                self.attns.append(DAttentionBaseline(fmap_size, fmap_size, heads, hc, n_groups, attn_drop, proj_drop, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, stage_idx))
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size))
            else:
                raise NotImplementedError(f'Spec={stage_spec[i]} is not supported.')
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)
        positions = []
        references = []
        for d in range(self.depths):
            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)
        return x, positions, references


class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4, dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], heads=[3, 6, 12, 24], window_sizes=[7, 7, 7, 7], drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, strides=[-1, -1, -1, -1], offset_range_factor=[1, 2, 3, 4], stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], groups=[-1, -1, 3, 6], use_pes=[False, False, False, False], dwc_pes=[False, False, False, False], sr_ratios=[8, 4, 2, 1], fixed_pes=[False, False, False, False], no_offs=[False, False, False, False], ns_per_pts=[4, 4, 4, 4], use_dwc_mlps=[False, False, False, False], use_conv_patches=False, **kwargs):
        super().__init__()
        self.patch_proj = nn.Sequential(nn.Conv2d(3, dim_stem, 7, patch_size, 3), LayerNormProxy(dim_stem)) if use_conv_patches else nn.Sequential(nn.Conv2d(3, dim_stem, patch_size, patch_size, 0), LayerNormProxy(dim_stem))
        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(TransformerStage(img_size, window_sizes[i], ns_per_pts[i], dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], sr_ratios[i], heads[i], strides[i], offset_range_factor[i], i, dwc_pes[i], no_offs[i], fixed_pes[i], attn_drop_rate, drop_rate, expansion, drop_rate, dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i]))
            img_size = img_size // 2
        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False), LayerNormProxy(dims[i + 1])) if use_conv_patches else nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False), LayerNormProxy(dims[i + 1])))
        self.cls_norm = LayerNormProxy(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict):
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        self.load_state_dict(new_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        x = self.patch_proj(x)
        positions = []
        references = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)
        return x, positions, references


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1, H=7, W=7, ratio=3, apply_transform=True):
        super(EMSA, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ratio = ratio
        if self.ratio > 1:
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2, groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)
        self.apply_transform = apply_transform and h > 1
        if self.apply_transform:
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq, c = queries.shape
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        if self.ratio > 1:
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)
            x = self.sr_conv(x)
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        if self.apply_transform:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)
            att = self.transform(att)
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)
            att = torch.softmax(att, -1)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)
        return out


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def to(x):
    return {'device': x.device, 'dtype': x.dtype}


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2
    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2
    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):

    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size
        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')
        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


class HaloAttention(nn.Module):

    def __init__(self, *, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.halo_size = halo_size
        inner_dim = dim_head * heads
        self.rel_pos_emb = RelPosEmb(block_size=block_size, rel_size=block_size + halo_size * 2, dim_head=dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)
        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c=c)
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))
        q *= self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)
        sim += self.rel_pos_emb(q)
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + halo * 2, stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=h // block, w=w // block, p1=block, p2=block)
        return out


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table, persistent=False)
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
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalAttention(nn.Module):
    """ MOA - multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, input_resolution, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.query_size = self.window_size[0]
        self.key_size = self.window_size[0] + 2
        h, w = input_resolution
        self.seq_len = h // self.query_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.reduction = 32
        self.pre_conv = nn.Conv2d(dim, int(dim // self.reduction), 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1) * (2 * self.seq_len - 1), num_heads))
        coords_h = torch.arange(self.seq_len)
        coords_w = torch.arange(self.seq_len)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.seq_len - 1
        relative_coords[:, :, 1] += self.seq_len - 1
        relative_coords[:, :, 0] *= 2 * self.seq_len - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.queryembedding = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=self.query_size, p2=self.query_size)
        self.keyembedding = nn.Unfold(kernel_size=(self.key_size, self.key_size), stride=14, padding=1)
        self.query_dim = int(dim // self.reduction) * self.query_size * self.query_size
        self.key_dim = int(dim // self.reduction) * self.key_size * self.key_size
        self.q = nn.Linear(self.query_dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.key_dim, 2 * self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, _, C = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.pre_conv(x)
        query = self.queryembedding(x).view(B, -1, self.query_dim)
        query = self.q(query)
        B, N, C = query.size()
        q = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.keyembedding(x).view(B, -1, self.key_dim)
        kv = self.kv(key).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = kv[0]
        v = kv[1]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.seq_len * self.seq_len, self.seq_len * self.seq_len, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


class LocalTransformerBlock(nn.Module):
    """ Local Transformer Block.

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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class WindowMultiHeadAttention(nn.Module):
    """This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.
    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int], drop_attn: float=0.0, drop_proj: float=0.0, meta_hidden_dim: int=384, sequential_attn: bool=False) ->None:
        super(WindowMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, 'The number of input features (in_features) are not divisible by the number of heads (num_heads).'
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        self.meta_mlp = Mlp(2, hidden_features=meta_hidden_dim, out_features=num_heads, act_layer=nn.ReLU, drop=(0.125, 0.0))
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) ->None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(torch.meshgrid([torch.arange(self.window_size[0], device=device), torch.arange(self.window_size[1], device=device)]), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(1.0 + relative_coordinates.abs())
        self.register_buffer('relative_coordinates_log', relative_coordinates_log, persistent=False)

    def update_input_size(self, new_window_size: int, **kwargs: Any) ->None:
        """Method updates the window size and so the pair-wise relative positions
        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        self.window_size: int = new_window_size
        self._make_pair_wise_relative_positions()

    def _relative_positional_encodings(self) ->torch.Tensor:
        """Method computes the relative positional encodings
        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(self.num_heads, window_area, window_area)
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def _forward_sequential(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        """
        assert False, 'not implemented'

    def _forward_batch(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """This function performs standard (non-sequential) scaled cosine self-attention.
        """
        Bw, L, C = x.shape
        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attn = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale
        attn = attn + self._relative_positional_encodings()
        if mask is not None:
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case
        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        if self.sequential_attn:
            return self._forward_sequential(x, mask)
        else:
            return self._forward_batch(x, mask)


class SwinTransformerBlock(nn.Module):
    """This class implements the Swin transformer block.
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized
    """

    def __init__(self, dim: int, num_heads: int, feat_size: Tuple[int, int], window_size: Tuple[int, int], shift_size: Tuple[int, int]=(0, 0), mlp_ratio: float=4.0, init_values: Optional[float]=0, drop: float=0.0, drop_attn: float=0.0, drop_path: float=0.0, extra_norm: bool=False, sequential_attn: bool=False, norm_layer: Type[nn.Module]=nn.LayerNorm) ->None:
        super(SwinTransformerBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values
        self.attn = WindowMultiHeadAttention(dim=dim, num_heads=num_heads, window_size=self.window_size, drop_attn=drop_attn, drop_proj=drop, sequential_attn=sequential_attn)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop, out_features=dim)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim) if extra_norm else nn.Identity()
        self._make_attention_mask()
        self.init_weights()

    def _calc_window_shift(self, target_window_size):
        window_size = [(f if f <= w else w) for f, w in zip(self.feat_size, target_window_size)]
        shift_size = [(0 if f <= w else s) for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) ->None:
        """Method generates the attention mask used in shift case."""
        if any(self.shift_size):
            H, W = self.feat_size
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for h in (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)):
                for w in (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def update_input_size(self, new_window_size: Tuple[int, int], new_feat_size: Tuple[int, int]) ->None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.
        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = new_feat_size
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(new_window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.update_input_size(new_window_size=self.window_size)
        self._make_attention_mask()

    def _shifted_window_attn(self, x):
        H, W = self.feat_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, self.feat_size)
        if do_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        x = x.view(B, L, C)
        return x

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, pretrained_window_size=pretrained_window_size) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class MOATransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_global = [x.item() for x in torch.linspace(0, 0.2, len(depths) - 1)]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, drop_path_global=dpr_global[i_layer] if i_layer < self.num_layers - 1 else 0, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
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
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class Depth_Pointwise_Conv1d(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, groups=1)

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)
        out = out + out2
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MobileViTAttention(nn.Module):

    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7, depth=3, mlp_dim=1024):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)
        self.trans = Transformer(dim=dim, depth=depth, heads=8, head_dim=64, mlp_dim=mlp_dim)
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()
        y = self.conv2(self.conv1(x))
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph, nw=w // self.pw)
        y = self.conv3(y)
        y = torch.cat([x, y], 1)
        y = self.conv4(y)
        return y


class MobileViTv2Attention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        """
        i = self.fc_i(input)
        weight_i = torch.softmax(i, dim=1)
        context_score = weight_i * self.fc_k(input)
        context_vector = torch.sum(context_score, dim=1, keepdim=True)
        v = self.fc_v(input) * context_vector
        out = self.fc_o(v)
        return out


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads, self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x


class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))
        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False), nn.Sigmoid()))
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        SPC_out = x.view(b, self.S, c // self.S, h, w)
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)
        softmax_out = self.softmax(SE_out)
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)
        return PSA_out


class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel, channel, kernel_size=1), nn.Sigmoid())
        self.conv1x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1), nn.BatchNorm2d(channel))
        self.conv3x3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel))
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x
        spatial_wv = self.sp_wv(x)
        spatial_wq = self.sp_wq(x)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x
        spatial_wv = self.sp_wv(channel_out)
        spatial_wq = self.sp_wq(channel_out)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * channel_out
        return spatial_out


class ResidualAttention(nn.Module):

    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y_raw = self.fc(x).flatten(2)
        y_avg = torch.mean(y_raw, dim=2)
        y_max = torch.max(y_raw, dim=2)[0]
        score = y_avg + self.la * y_max
        return score


class SplitAttention(nn.Module):

    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-05
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)), ('bn', nn.BatchNorm2d(channel)), ('relu', nn.ReLU())])))
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))
        attention_weughts = torch.stack(weights, 0)
        attention_weughts = self.softmax(attention_weughts)
        V = (attention_weughts * feats).sum(0)
        return V


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class SimAM(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=0.0001):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'lambda=%f)' % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return 'simam'

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):

    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):

    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor


class UFOAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        kv = torch.matmul(k, v)
        kv_norm = XNorm(kv, self.gamma)
        q_norm = XNorm(q, self.gamma)
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MLP(nn.Module):
    """
    Build a Multi-Layer Perceptron
        - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, planes, mlp_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(planes, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, planes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WeightedPermuteMLP(nn.Module):

    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.segment_dim = segment_dim
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GFNet(nn.Module):

    def __init__(self, embed_dim=384, img_size=224, patch_size=16, mlp_ratio=4, depth=4, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.embedding = nn.Linear(patch_size ** 2 * 3, embed_dim)
        h = img_size // patch_size
        w = h // 2 + 1
        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, h=h, w=w) for i in range(depth)])
        self.head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.softmax(self.head(x))
        return x


class CMT(nn.Module):

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[46, 92, 184, 368], stem_channel=16, fc_dim=1280, num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, depths=[2, 2, 10, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-05)
        self.patch_embed_a = PatchEmbed(img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.relative_pos_a = nn.Parameter(torch.randn(num_heads[0], self.patch_embed_a.num_patches, self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(num_heads[1], self.patch_embed_b.num_patches, self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(num_heads[2], self.patch_embed_c.num_patches, self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(num_heads[3], self.patch_embed_d.num_patches, self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_a = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-05)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)
        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)
        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PyramidVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // 2 ** (i + 1), patch_size=patch_size if i == 0 else 2, in_chans=in_chans if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            block = nn.ModuleList([Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, sr_ratio=sr_ratios[i]) for j in range(depths[i])])
            cur += depths[i]
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'pos_embed{i + 1}', pos_embed)
            setattr(self, f'pos_drop{i + 1}', pos_drop)
            setattr(self, f'block{i + 1}', block)
        self.norm = norm_layer(embed_dims[3])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        for i in range(num_stages):
            pos_embed = getattr(self, f'pos_embed{i + 1}')
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2), size=(H, W), mode='bilinear').reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            pos_embed = getattr(self, f'pos_embed{i + 1}')
            pos_drop = getattr(self, f'pos_drop{i + 1}')
            block = getattr(self, f'block{i + 1}')
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PosCNN(nn.Module):

    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return [('proj.%d.weight' % i) for i in range(4)]


class Class_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class LayerScale_Block_CA(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention, Mlp_block=Mlp, init_values=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Attention_talking_head(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention_talking_head, Mlp_block=Mlp, init_values=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaiT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, global_pool=None, block_layers=LayerScale_Block, block_layers_token=LayerScale_Block_CA, Patch_layer=PatchEmbed, act_layer=nn.GELU, Attention_block=Attention_talking_head, Mlp_block=Mlp, init_scale=0.0001, Attention_block_token_only=Class_Attention, Mlp_block_token_only=Mlp, depth_token_only=2, mlp_ratio_clstk=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([block_layers(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale) for i in range(depth)])
        self.blocks_token_only = nn.ModuleList([block_layers_token(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer, act_layer=act_layer, Attention_block=Attention_block_token_only, Mlp_block=Mlp_block_token_only, init_values=init_scale) for i in range(depth_token_only)])
        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Image2Tokens(nn.Module):

    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super(Image2Tokens, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


class LocallyEnhancedFeedForward(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=hidden_features)
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()
        cls_token, tokens = torch.split(x, [1, n - 1], dim=1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out


class AttentionLCA(Attention):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super(AttentionLCA, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.qkv_bias = qkv_bias

    def forward(self, x):
        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]
        B, N, C = x.shape
        _, last_token = torch.split(x, [N - 1, 1], dim=1)
        q = F.linear(last_token, q_weight, q_bias).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = F.linear(x, kv_weight, kv_bias).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridEmbed(nn.Module):
    """ 
        Same embedding as timm lib.
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


class CeIT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, leff_local_size=3, leff_with_bn=True):
        """
        args:
            - img_size (:obj:`int`): input image size
            - patch_size (:obj:`int`): patch size
            - in_chans (:obj:`int`): input channels
            - num_classes (:obj:`int`): number of classes
            - embed_dim (:obj:`int`): embedding dimensions for tokens
            - depth (:obj:`int`): depth of encoder
            - num_heads (:obj:`int`): number of heads in multi-head self-attention
            - mlp_ratio (:obj:`float`): expand ratio in feedforward
            - qkv_bias (:obj:`bool`): whether to add bias for mlp of qkv
            - qk_scale (:obj:`float`): scale ratio for qk, default is head_dim ** -0.5
            - drop_rate (:obj:`float`): dropout rate in feedforward module after linear operation
                and projection drop rate in attention
            - attn_drop_rate (:obj:`float`): dropout rate for attention
            - drop_path_rate (:obj:`float`): drop_path rate after attention
            - hybrid_backbone (:obj:`nn.Module`): backbone e.g. resnet
            - norm_layer (:obj:`nn.Module`): normalization type
            - leff_local_size (:obj:`int`): kernel size in LocallyEnhancedFeedForward
            - leff_with_bn (:obj:`bool`): whether add bn in LocallyEnhancedFeedForward
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.i2t = HybridEmbed(hybrid_backbone, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.i2t.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, kernel_size=leff_local_size, with_bn=leff_with_bn) for i in range(depth)])
        self.lca = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer, feedforward_type='lca')
        self.pos_layer_embed = nn.Parameter(torch.zeros(1, depth, embed_dim))
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.i2t(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        cls_token_list = []
        for blk in self.blocks:
            x, curr_cls_token = blk(x)
            cls_token_list.append(curr_cls_token)
        all_cls_token = torch.stack(cls_token_list, dim=1)
        all_cls_token = all_cls_token + self.pos_layer_embed
        last_cls_token = self.lca(all_cls_token)
        last_cls_token = self.norm(last_cls_token)
        return last_cls_token.view(B, -1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()
        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch, kernel_size=(cur_window, cur_window), padding=(padding_size, padding_size), dilation=(dilation, dilation), groups=cur_head_split * Ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [(x * Ch) for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W
        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)
        EV_hat_img = q_img * conv_v_img
        zero = torch.zeros((B, h, 1, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding. 
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W
        cls_token, img_tokens = x[:, :1], x[:, 1:]
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()
        self.cpe = shared_cpe
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)
        return x


class ParallelBlock(nn.Module):
    """ Parallel block class. """

    def __init__(self, dims, num_heads, mlp_ratios=[], qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpes=None, shared_crpes=None):
        super().__init__()
        self.cpes = shared_cpes
        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[1])
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[2])
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[3])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def upsample(self, x, output_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W
        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]
        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(img_tokens, size=output_size, mode='bilinear')
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)
        out = torch.cat((cls_token, img_tokens), dim=1)
        return out

    def forward(self, x1, x2, x3, x4, sizes):
        _, (H2, W2), (H3, W3), (H4, W4) = sizes
        x2 = self.cpes[1](x2, size=(H2, W2))
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=(H2, W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3, W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4, W4))
        upsample3_2 = self.upsample(cur3, output_size=(H2, W2), size=(H3, W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3, W3), size=(H4, W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2, W2), size=(H4, W4))
        downsample2_3 = self.downsample(cur2, output_size=(H3, W3), size=(H2, W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4, W4), size=(H3, W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4, W4), size=(H2, W2))
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        return x1, x2, x3, x4


class CoaT(nn.Module):
    """ CoaT class. """

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[0, 0, 0, 0], serial_depths=[0, 0, 0, 0], parallel_depth=0, num_heads=0, mlp_ratios=[0, 0, 0, 0], qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), return_interm_layers=False, out_features=None, crpe_window={(3): 2, (5): 3, (7): 3}, **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        dpr = drop_path_rate
        self.serial_blocks1 = nn.ModuleList([SerialBlock(dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe1, shared_crpe=self.crpe1) for _ in range(serial_depths[0])])
        self.serial_blocks2 = nn.ModuleList([SerialBlock(dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe2, shared_crpe=self.crpe2) for _ in range(serial_depths[1])])
        self.serial_blocks3 = nn.ModuleList([SerialBlock(dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe3, shared_crpe=self.crpe3) for _ in range(serial_depths[2])])
        self.serial_blocks4 = nn.ModuleList([SerialBlock(dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe4, shared_crpe=self.crpe4) for _ in range(serial_depths[3])])
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([ParallelBlock(dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4], shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4]) for _ in range(parallel_depth)])
        if not self.return_interm_layers:
            self.norm1 = norm_layer(embed_dims[0])
            self.norm2 = norm_layer(embed_dims[1])
            self.norm3 = norm_layer(embed_dims[2])
            self.norm4 = norm_layer(embed_dims[3])
            if self.parallel_depth > 0:
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
                self.head = nn.Linear(embed_dims[3], num_classes)
            else:
                self.head = nn.Linear(embed_dims[3], num_classes)
        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        trunc_normal_(self.cls_token3, std=0.02)
        trunc_normal_(self.cls_token4, std=0.02)
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
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def remove_cls(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :]

    def forward_features(self, x0):
        B = x0.shape[0]
        x1, (H1, W1) = self.patch_embed1(x0)
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_cls(x1)
        x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2, (H2, W2) = self.patch_embed2(x1_nocls)
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_cls(x2)
        x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3, (H3, W3) = self.patch_embed3(x2_nocls)
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_cls(x3)
        x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x4, (H4, W4) = self.patch_embed4(x3_nocls)
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_cls(x4)
        x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        if self.parallel_depth == 0:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                x4_cls = x4[:, 0]
                return x4_cls
        for blk in self.parallel_blocks:
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])
        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = self.remove_cls(x1)
                x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = self.remove_cls(x2)
                x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = self.remove_cls(x3)
                x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = self.remove_cls(x4)
                x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            merged_cls = torch.cat((x2_cls, x3_cls, x4_cls), dim=1)
            merged_cls = self.aggregate(merged_cls).squeeze(dim=1)
            return merged_cls

    def forward(self, x):
        if self.return_interm_layers:
            return self.forward_features(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


class ConvBN(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, bn=True):
        padding = (kernel_size - 1) // 2
        if bn:
            super(ConvBN, self).__init__(OrderedDict([('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False)), ('bn', nn.BatchNorm2d(out_planes))]))
        else:
            super(ConvBN, self).__init__(OrderedDict([('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False))]))


class MHSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = q @ k.transpose(-2, -1) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)
        img_size = int(N ** 0.5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** 0.5
        distances = distances
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        if return_map:
            return dist, attn_map
        else:
            return dist

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
        return x


class STE(nn.Module):
    """
    Build a Standard Transformer Encoder(STE)
    input: Tensor (b, c, h, w)
    output: Tensor (b, c, h, w)
    """

    def __init__(self, planes: int, mlp_dim: int, head_num: int, dropout: float, patch_size: int, relative: bool, qkv_bias: bool, pre_norm: bool, **kwargs):
        super(STE, self).__init__()
        self.patch_size = patch_size
        self.pre_norm = pre_norm
        self.relative = relative
        self.flatten = nn.Sequential(Rearrange('b c pnh pnw psh psw -> (b pnh pnw) psh psw c'))
        if not relative:
            self.pe = nn.ParameterList([nn.Parameter(torch.zeros(1, patch_size, 1, planes // 2)), nn.Parameter(torch.zeros(1, 1, patch_size, planes // 2))])
        self.attn = MHSA(planes, head_num, dropout, patch_size, qkv_bias=qkv_bias, relative=relative)
        self.mlp = MLP(planes, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(planes)
        self.norm2 = nn.LayerNorm(planes)

    def forward(self, x):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        patch_num_h, patch_num_w = h // patch_size, w // patch_size
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = self.flatten(x)
        if not self.relative:
            x_h, x_w = x.split(c // 2, dim=3)
            x = torch.cat((x_h + self.pe[0], x_w + self.pe[1]), dim=3)
        x = rearrange(x, 'b psh psw c -> b (psh psw) c')
        if self.pre_norm:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))
        x = rearrange(x, '(b pnh pnw) (psh psw) c -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size)
        return x


class ConTBlock(nn.Module):
    """
    Build a ConTBlock
    """

    def __init__(self, planes: int, out_planes: int, mlp_dim: int, head_num: int, dropout: float, patch_size: List[int], downsample: nn.Module=None, stride: int=1, last_dropout: float=0.3, **kwargs):
        super(ConTBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Identity()
        self.dropout = nn.Identity()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ste1 = STE(planes=planes, mlp_dim=mlp_dim, head_num=head_num, dropout=dropout, patch_size=patch_size[0], **kwargs)
        self.ste2 = STE(planes=planes, mlp_dim=mlp_dim, head_num=head_num, dropout=dropout, patch_size=patch_size[1], **kwargs)
        if stride == 1 and downsample is not None:
            self.dropout = nn.Dropout(p=last_dropout)
            kernel_size = 1
        else:
            kernel_size = 3
        self.out_conv = ConvBN(planes, out_planes, kernel_size, stride, bn=False)

    def forward(self, x):
        x_preact = self.relu(self.bn(x))
        identity = self.identity(x)
        if self.downsample is not None:
            identity = self.downsample(x_preact)
        residual = self.ste1(x_preact)
        residual = self.ste2(residual)
        residual = self.out_conv(residual)
        out = self.dropout(residual + identity)
        return out


class ConTNet(nn.Module):
    """
    Build a ConTNet backbone
    """

    def __init__(self, block, layers: List[int], mlp_dim: List[int], head_num: List[int], dropout: List[float], in_channels: int=3, inplanes: int=64, num_classes: int=1000, init_weights: bool=True, first_embedding: bool=False, tweak_C: bool=False, **kwargs):
        """
        Args:
            block: ConT Block
            layers: number of blocks at each layer
            mlp_dim: dimension of mlp in each stage
            head_num: number of head in each stage
            dropout: dropout in the last two stage
            relative: if True, relative Position Embedding is used
            groups: nunmber of group at each conv layer in the Network
            depthwise: if True, depthwise convolution is adopted
            in_channels: number of channels of input image
            inplanes: channel of the first convolution layer
            num_classes: number of classes for classification task
                         only useful when `with_classifier` is True
            with_avgpool: if True, an average pooling is added at the end of resnet stage5
            with_classifier: if True, FC layer is registered for classification task
            first_embedding: if True, a conv layer with both stride and kernel of 7 is placed at the top
            tweakC: if true, the first layer of ResNet-C replace the ori layer
        """
        super(ConTNet, self).__init__()
        self.inplanes = inplanes
        self.block = block
        if tweak_C:
            self.layer0 = nn.Sequential(OrderedDict([('conv_bn1', ConvBN(in_channels, inplanes // 2, kernel_size=3, stride=2)), ('relu1', nn.ReLU(inplace=True)), ('conv_bn2', ConvBN(inplanes // 2, inplanes // 2, kernel_size=3, stride=1)), ('relu2', nn.ReLU(inplace=True)), ('conv_bn3', ConvBN(inplanes // 2, inplanes, kernel_size=3, stride=1)), ('relu3', nn.ReLU(inplace=True)), ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        elif first_embedding:
            self.layer0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, inplanes, kernel_size=4, stride=4)), ('norm', nn.LayerNorm(inplanes))]))
        else:
            self.layer0 = nn.Sequential(OrderedDict([('conv', ConvBN(in_channels, inplanes, kernel_size=7, stride=2, bn=False)), ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        self.cont_layers = []
        self.out_channels = OrderedDict()
        for i in range(len(layers)):
            stride = 2,
            patch_size = [7, 14]
            if i == len(layers) - 1:
                stride, patch_size[1] = 1, 7
            cont_layer = self._make_layer(inplanes * 2 ** i, layers[i], stride=stride, mlp_dim=mlp_dim[i], head_num=head_num[i], dropout=dropout[i], patch_size=patch_size, **kwargs)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, cont_layer)
            self.cont_layers.append(layer_name)
            self.out_channels[layer_name] = 2 * inplanes * 2 ** i
        self.last_out_channels = next(reversed(self.out_channels.values()))
        self.fc = nn.Linear(self.last_out_channels, num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int, mlp_dim: int, head_num: int, dropout: float, patch_size: List[int], use_avgdown: bool=False, **kwargs):
        layers = OrderedDict()
        for i in range(0, blocks - 1):
            layers[f'{self.block.__name__}{i}'] = self.block(planes, planes, mlp_dim, head_num, dropout, patch_size, **kwargs)
        downsample = None
        if stride != 1:
            if use_avgdown:
                downsample = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)), ('conv', ConvBN(planes, planes * 2, kernel_size=1, stride=1, bn=False))]))
            else:
                downsample = ConvBN(planes, planes * 2, kernel_size=1, stride=2, bn=False)
        else:
            downsample = ConvBN(planes, planes * 2, kernel_size=1, stride=1, bn=False)
        layers[f'{self.block.__name__}{blocks - 1}'] = self.block(planes, planes * 2, mlp_dim, head_num, dropout, patch_size, downsample, stride, **kwargs)
        return nn.Sequential(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        for _, layer_name in enumerate(self.cont_layers):
            cont_layer = getattr(self, layer_name)
            x = cont_layer(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


class GPSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, locality_strength=1.0, use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1) != N:
            self.get_rel_indices(N)
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = q @ k.transpose(-2, -1) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)
        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)
        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.0):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1
        kernel_size = int(self.num_heads ** 0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** 0.5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        self.rel_indices = rel_indices


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))
        if len(self.blocks) == 0:
            self.blocks = None
        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d + 1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))
        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))
        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d + 1) % num_branches]), act_layer(), nn.Linear(dim[(d + 1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [(i // p * i // p) for i, p in zip(img_size, patches)]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_heads=(6, 12), mlp_ratio=(2.0, 2.0, 4.0), qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()
        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size
        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)
        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))
            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([(nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()) for i in range(self.num_branches)])
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=0.02)
            trunc_normal_(self.cls_token[i], std=0.02)
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
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)
        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        return out

    def forward(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        return ce_logits


class CMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_pure(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class MixBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, 3 * dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.attn = Attention_pure(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sa_weight = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, _, H, W = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.conv1(x)
        conv = qkv[:, 2 * self.dim:, :, :]
        conv = self.conv(conv)
        sa = qkv.flatten(2).transpose(1, 2)
        sa = self.attn(sa)
        sa = sa.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = residual + self.drop_path(self.conv2(torch.sigmoid(self.sa_weight) * sa + (1 - torch.sigmoid(self.sa_weight)) * conv))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ReAttention(nn.Module):
    """
    It is observed that similarity along same batch of data is extremely large. 
    Thus can reduce the bs dimension when calculating the attention map.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, expansion_ratio=3, apply_transform=True, transform_scale=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


class PatchEmbed_CNN(nn.Module):
    """ 
        Following T2T, we use 3 layers of CNN for comparison with other methods.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, spp=32):
        super().__init__()
        new_patch_size = to_2tuple(patch_size // 2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DeepVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, group=False, re_atten=True, cos_reg=False, use_cnn_embed=False, apply_transform=None, transform_scale=False, scale_adjustment=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.cos_reg = cos_reg
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        elif use_cnn_embed:
            self.patch_embed = PatchEmbed_CNN(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        d = depth if isinstance(depth, int) else len(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, d)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, share=depth[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group, re_atten=re_atten, apply_transform=apply_transform[i], transform_scale=transform_scale, scale_adjustment=scale_adjustment) for i in range(len(depth))])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.cos_reg:
            atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if self.cos_reg:
                atten_list.append(attn)
        x = self.norm(x)
        if self.cos_reg and self.training:
            return x[:, 0], atten_list
        else:
            return x[:, 0]

    def forward(self, x):
        if self.cos_reg and self.training:
            x, atten = self.forward_features(x)
            x = self.head(x)
            return x, atten
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


class DistilledVisionTransformer(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


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


def meta_blocks(dim, index, layers, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, vit_num=1):
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(Meta3D(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
        else:
            blocks.append(Meta4D(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())
    blocks = nn.Sequential(*blocks)
    return blocks


def stem(in_chs, out_chs):
    return nn.Sequential(nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_chs // 2), nn.ReLU(), nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_chs), nn.ReLU())


class EfficientFormer(nn.Module):

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None, pool_size=3, norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, fork_feat=False, init_cfg=None, pretrained=None, vit_num=0, distillation=True, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratios, act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(Embedding(patch_size=down_patch_size, stride=down_stride, padding=down_pad, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
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
            cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        return cls_out


class InvertedResidual(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3, drop=0.0, act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(nn.GroupNorm(1, in_dim, eps=1e-06), nn.Conv2d(in_dim, hidden_dim, 1, bias=False), act_layer(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False), act_layer(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(hidden_dim, out_dim, 1, bias=False), nn.GroupNorm(1, out_dim, eps=1e-06))
        self.drop = nn.Dropout2d(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class HATNet(nn.Module):

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, dims=[64, 128, 256, 512], head_dim=64, expansions=[4, 4, 6, 6], grid_sizes=[1, 1, 1, 1], ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], drop_rate=0.0, drop_path_rate=0.0, act_layer=nn.SiLU, kernel_sizes=[3, 3, 3, 3]):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2), nn.GroupNorm(1, 16, eps=1e-06), act_layer(inplace=True), nn.Conv2d(16, dims[0], 3, padding=1, stride=2))
        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage], expansion=expansions[stage], drop=drop_rate, drop_path=dpr[sum(depths[:stage]) + i], kernel_size=kernel_sizes[stage], act_layer=act_layer) for i in range(depths[stage])]))
        self.blocks = nn.ModuleList(self.blocks)
        self.ds2 = Downsample(dims[0], dims[1])
        self.ds3 = Downsample(dims[1], dims[2])
        self.ds4 = Downsample(dims[2], dims[3])
        self.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(dims[-1], num_classes))
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for stage in range(len(self.blocks)):
            for idx in range(self.depths[stage]):
                self.blocks[stage][idx].drop_path.drop_prob = dpr[cur + idx]
            cur += self.depths[stage]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)
        for block in self.blocks[3]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.classifier(x)
        return x


class Conv2d_BN(torch.nn.Sequential):

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)
        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation * (ks - 1) - 1) // stride + 1) ** 2
        FLOPS_COUNTER += a * b * output_points * ks ** 2 // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):

    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)
        global FLOPS_COUNTER
        output_points = resolution ** 2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Subsample(torch.nn.Module):

    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[:, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):

    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2, activation=None, stride=2, resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)
        self.q = torch.nn.Sequential(Subsample(stride, resolution), Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(self.dh, out_dim, resolution=resolution_))
        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N_, N))
        global FLOPS_COUNTER
        FLOPS_COUNTER += num_heads * resolution ** 2 * resolution_ ** 2 * key_dim
        FLOPS_COUNTER += num_heads * resolution ** 2 * resolution_ ** 2
        FLOPS_COUNTER += num_heads * resolution ** 2 * resolution_ ** 2 * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=[192], key_dim=[64], depth=[12], num_heads=[3], attn_ratio=[2], mlp_ratio=[2], hybrid_backbone=None, down_ops=[], attention_activation=torch.nn.Hardswish, mlp_activation=torch.nn.Hardswish, distillation=True, drop_path=0):
        super().__init__()
        global FLOPS_COUNTER
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(Residual(Attention(ed, kd, nh, attn_ratio=ar, activation=attention_activation, resolution=resolution), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(Residual(torch.nn.Sequential(Linear_BN(ed, h, resolution=resolution), mlp_activation(), Linear_BN(h, ed, bn_weight_init=0, resolution=resolution)), drop_path))
            if do[0] == 'Subsample':
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(AttentionSubsample(*embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2], attn_ratio=do[3], activation=attention_activation, stride=do[5], resolution=resolution, resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(Residual(torch.nn.Sequential(Linear_BN(embed_dim[i + 1], h, resolution=resolution), mlp_activation(), Linear_BN(h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution)), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


class MV2Block(nn.Module):

    def __init__(self, inp, out, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expansion
        self.use_res_connection = stride == 1 and inp == out
        if expansion == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, bias=False), nn.SiLU(), nn.BatchNorm2d(out))

    def forward(self, x):
        if self.use_res_connection:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        return out


def conv_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2), nn.BatchNorm2d(oup), nn.SiLU())


class MobileViT(nn.Module):

    def __init__(self, image_size, dims, channels, num_classes, depths=[2, 4, 3], expansion=4, kernel_size=3, patch_size=2):
        super().__init__()
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert iw % pw == 0 and ih % ph == 0
        self.conv1 = conv_bn(3, channels[0], kernel_size=3, stride=patch_size)
        self.mv2 = nn.ModuleList([])
        self.m_vits = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1))
        self.mv2.append(MV2Block(channels[1], channels[2], 2))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[3], channels[4], 2))
        self.m_vits.append(MobileViTAttention(channels[4], dim=dims[0], kernel_size=kernel_size, patch_size=patch_size, depth=depths[0], mlp_dim=int(2 * dims[0])))
        self.mv2.append(MV2Block(channels[4], channels[5], 2))
        self.m_vits.append(MobileViTAttention(channels[5], dim=dims[1], kernel_size=kernel_size, patch_size=patch_size, depth=depths[1], mlp_dim=int(4 * dims[1])))
        self.mv2.append(MV2Block(channels[5], channels[6], 2))
        self.m_vits.append(MobileViTAttention(channels[6], dim=dims[2], kernel_size=kernel_size, patch_size=patch_size, depth=depths[2], mlp_dim=int(4 * dims[2])))
        self.conv2 = conv_bn(channels[-2], channels[-1], kernel_size=1)
        self.pool = nn.AvgPool2d(image_size // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.mv2[0](y)
        y = self.mv2[1](y)
        y = self.mv2[2](y)
        y = self.mv2[3](y)
        y = self.mv2[4](y)
        y = self.m_vits[0](y)
        y = self.mv2[5](y)
        y = self.m_vits[1](y)
        y = self.mv2[6](y)
        y = self.m_vits[2](y)
        y = self.conv2(y)
        y = self.pool(y).view(y.shape[0], -1)
        y = self.fc(y)
        return y


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride, padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class conv_embedding(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):

    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads, mlp_ratio, num_classes=1000, in_chans=3, attn_drop_rate=0.0, drop_rate=0.0, drop_path_rate=0.0):
        super(PoolingTransformer, self).__init__()
        total_block = sum(depth)
        padding = 0
        block_idx = 0
        width = math.floor((image_size + 2 * padding - patch_size) / stride + 1)
        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], width, width), requires_grad=True)
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, base_dims[0] * heads[0]), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        for stage in range(len(depth)):
            drop_path_prob = [(drop_path_rate * i / total_block) for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]
            self.transformers.append(Transformer(base_dims[stage], depth[stage], heads[stage], mlp_ratio, drop_rate, attn_drop_rate, drop_path_prob))
            if stage < len(heads) - 1:
                self.pools.append(conv_head_pooling(base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2))
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-06)
        self.embed_dim = base_dims[-1] * heads[-1]
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 2, self.base_dims[0] * self.heads[0]), requires_grad=True)
        if self.num_classes > 0:
            self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1], self.num_classes)
        else:
            self.head_dist = nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2


class Learned_Aggregation_Layer(nn.Module):

    def __init__(self, dim: int, num_heads: int=1, qkv_bias: bool=False, qk_scale: Optional[float]=None, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class Learned_Aggregation_Layer_multi(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=False, qk_scale: Optional[float]=None, attn_drop: float=0.0, proj_drop: float=0.0, num_classes: int=1000):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, :self.num_classes]).reshape(B, self.num_classes, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x[:, self.num_classes:]).reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x[:, self.num_classes:]).reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_classes, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class Layer_scale_init_Block_only_token(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, qk_scale: Optional[float]=None, drop: float=0.0, attn_drop: float=0.0, drop_path: float=0.0, act_layer: nn.Module=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Learned_Aggregation_Layer, Mlp_block=Mlp, init_values: float=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor) ->torch.Tensor:
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Conv_blocks_se(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.qkv_pos = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.Conv2d(dim, dim, groups=dim, kernel_size=3, padding=1, stride=1, bias=True), nn.GELU(), SqueezeExcite(dim, rd_ratio=0.25), nn.Conv2d(dim, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(-1, -2)
        x = x.reshape(B, C, H, W)
        x = self.qkv_pos(x)
        x = x.reshape(B, C, N)
        x = x.transpose(-1, -2)
        return x


class Layer_scale_init_Block(nn.Module):

    def __init__(self, dim: int, drop_path: float=0.0, act_layer: nn.Module=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=None, init_values: float=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))


def conv3x3(in_planes: int, out_planes: int, stride: int=1) ->nn.Sequential:
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


class ConvStem(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int=224, patch_size: int=16, in_chans: int=3, embed_dim: int=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Sequential(conv3x3(in_chans, embed_dim // 8, 2), nn.GELU(), conv3x3(embed_dim // 8, embed_dim // 4, 2), nn.GELU(), conv3x3(embed_dim // 4, embed_dim // 2, 2), nn.GELU(), conv3x3(embed_dim // 2, embed_dim, 2))

    def forward(self, x: torch.Tensor, padding_size: Optional[int]=None) ->torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchConvnet(nn.Module):

    def __init__(self, img_size: int=224, patch_size: int=16, in_chans: int=3, num_classes: int=1000, embed_dim: int=768, depth: int=12, num_heads: int=1, qkv_bias: bool=False, qk_scale: Optional[float]=None, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer=nn.LayerNorm, global_pool: Optional[str]=None, block_layers=Layer_scale_init_Block, block_layers_token=Layer_scale_init_Block_only_token, Patch_layer=ConvStem, act_layer: nn.Module=nn.GELU, Attention_block=Conv_blocks_se, dpr_constant: bool=True, init_scale: float=0.0001, Attention_block_token_only=Learned_Aggregation_Layer, Mlp_block_token_only=Mlp, depth_token_only: int=1, mlp_ratio_clstk: float=3.0, multiclass: bool=False):
        super().__init__()
        self.multiclass = multiclass
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if not self.multiclass:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, num_classes, int(embed_dim)))
        if not dpr_constant:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([block_layers(dim=embed_dim, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, Attention_block=Attention_block, init_values=init_scale) for i in range(depth)])
        self.blocks_token_only = nn.ModuleList([block_layers_token(dim=int(embed_dim), num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer, act_layer=act_layer, Attention_block=Attention_block_token_only, Mlp_block=Mlp_block_token_only, init_values=init_scale) for i in range(depth_token_only)])
        self.norm = norm_layer(int(embed_dim))
        self.total_len = depth_token_only + depth
        self.feature_info = [dict(num_chs=int(embed_dim), reduction=0, module='head')]
        if not self.multiclass:
            self.head = nn.Linear(int(embed_dim), num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.ModuleList([nn.Linear(int(embed_dim), 1) for _ in range(num_classes)])
        self.rescale: float = 0.02
        trunc_normal_(self.cls_token, std=self.rescale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes: int, global_pool: str=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) ->torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        if not self.multiclass:
            return x[:, 0]
        else:
            return x[:, :self.num_classes].reshape(B, self.num_classes, -1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B = x.shape[0]
        x = self.forward_features(x)
        if not self.multiclass:
            x = self.head(x)
            return x
        else:
            all_results = []
            for i in range(self.num_classes):
                all_results.append(self.head[i](x[:, i]))
            return torch.cat(all_results, dim=1).reshape(B, self.num_classes)


class StageModule(nn.Module):

    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, shuffle=True, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, relative_pos_embedding=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim)
        else:
            self.patch_partition = None
        num = layers // 2
        self.layers = nn.ModuleList([])
        for idx in range(num):
            the_last = idx == num - 1
            self.layers.append(nn.ModuleList([Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, relative_pos_embedding=relative_pos_embedding), Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, relative_pos_embedding=relative_pos_embedding)]))

    def forward(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x


class PatchEmbedding(nn.Module):

    def __init__(self, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(inter_channel), nn.ReLU6(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x


class ShuffleTransformer(nn.Module):

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, token_dim=32, embed_dim=96, mlp_ratio=4.0, layers=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], relative_pos_embedding=True, shuffle=True, window_size=7, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, has_pos_embed=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.has_pos_embed = has_pos_embed
        dims = [(i * 32) for i in num_heads]
        self.to_token = PatchEmbedding(inter_channel=token_dim, out_channels=embed_dim)
        num_patches = img_size * img_size // 16
        if self.has_pos_embed:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
            self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.stage1 = StageModule(layers[0], embed_dim, dims[0], num_heads[0], window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3], relative_pos_embedding=relative_pos_embedding)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.to_token(x)
        b, c, h, w = x.shape
        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SE(nn.Module):

    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, dim), nn.Tanh())

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)
        a = self.fc(a)
        x = a * x
        return x


class TNT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, outer_dim=768, inner_dim=48, depth=12, outer_num_heads=12, inner_num_heads=4, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, inner_stride=4, se=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim, inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)
        self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.outer_pos, std=0.02)
        trunc_normal_(self.inner_pos, std=0.02)
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
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos
        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)
        outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Outlooker(nn.Module):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """

    def __init__(self, dim, kernel_size, padding, stride=1, num_heads=1, mlp_ratio=3.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size, padding=padding, stride=stride, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = q * self.scale @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    """
    get block by name, specifically for class attention block in here
    """
    if block_type == 'ca':
        return ClassBlock(**kargs)


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3, padding=1, stride=1, mlp_ratio=3.0, qkv_bias=False, qk_scale=None, attn_drop=0, drop_path_rate=0.0, **kwargs):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding, stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop_path=block_dpr))
    blocks = nn.Sequential(*blocks)
    return blocks


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.0, qkv_bias=False, qk_scale=None, attn_drop=0, drop_path_rate=0.0, **kwargs):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop_path=block_dpr))
    blocks = nn.Sequential(*blocks)
    return blocks


class VOLO(nn.Module):
    """
    Vision Outlooker, the main class of our model
    --layers: [x,x,x,x], four blocks in two stages, the first block is outlooker, the
              other three are transformer, we set four blocks, which are easily
              applied to downstream tasks
    --img_size, --in_chans, --num_classes: these three are very easy to understand
    --patch_size: patch_size in outlook attention
    --stem_hidden_dim: hidden dim of patch embedding, d1-d4 is 64, d5 is 128
    --embed_dims, --num_heads: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios, --qkv_bias, --qk_scale, --drop_rate: easy to undertand
    --attn_drop_rate, --drop_path_rate, --norm_layer: easy to undertand
    --post_layers: post layers like two class attention layers using [ca, ca],
                  if yes, return_mean=False
    --return_mean: use mean of all feature tokens for classification, if yes, no class token
    --return_dense: use token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --mix_token: mixing tokens as token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --pooling_scale: pooling_scale=2 means we downsample 2x
    --out_kernel, --out_stride, --out_padding: kerner size,
                                               stride, and padding for outlook attention
    """

    def __init__(self, layers, img_size=224, in_chans=3, num_classes=1000, patch_size=8, stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=None, outlook_attention=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, post_layers=None, return_mean=False, return_dense=True, mix_token=True, pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size, in_chans=in_chans, hidden_dim=stem_hidden_dim, embed_dim=embed_dims[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size // pooling_scale, img_size // patch_size // pooling_scale, embed_dims[-1]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers, downsample=downsamples[i], num_heads=num_heads[i], kernel_size=out_kernel, stride=out_stride, padding=out_padding, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            else:
                stage = transformer_blocks(Transformer, i, embed_dims[i], layers, num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path_rate=drop_path_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            if downsamples[i]:
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))
        self.network = nn.ModuleList(network)
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([get_block(post_layers[i], dim=embed_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratios[-1], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer) for i in range(len(post_layers))])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=0.02)
        self.return_mean = return_mean
        self.return_dense = return_dense
        if return_dense:
            assert not return_mean, 'cannot return both mean and dense'
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:
            self.beta = 1.0
            assert return_dense, 'return all tokens if mix_token is enabled'
        if return_dense:
            self.aux_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
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
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            if idx == 2:
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale * bbx1, self.pooling_scale * bby1, self.pooling_scale * bbx2, self.pooling_scale * bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        if self.return_mean:
            return self.head(x.mean(1))
        x_cls = self.head(x[:, 0])
        if not self.return_dense:
            return x_cls
        x_aux = self.aux_head(x[:, 1:])
        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]
        if self.mix_token and self.training:
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, in_channel, channel, stride=1, C=32, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, stride=1, groups=C)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)
        self.relu = nn.ReLU(False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        if self.downsample != None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            self.downsample = nn.Conv2d(self.in_channel, channel * block.expansion, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(block(self.in_channel, channel, downsample=self.downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024 * block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            self.downsample = nn.Conv2d(self.in_channel, channel * block.expansion, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(block(self.in_channel, channel, downsample=self.downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None, window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init='', **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i in range(self.num_layers):
            layers += [BasicLayer(dim=embed_dim[i], out_dim=embed_out_dim[i], input_resolution=(self.patch_grid[0] // 2 ** i, self.patch_grid[1] // 2 ** i), depth=depths[i], num_heads=num_heads[i], head_dim=head_dim[i], window_size=window_size[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, downsample=PatchMerging if i < self.num_layers - 1 else None)]
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^absolute_pos_embed|patch_embed', blocks='^layers\\.(\\d+)' if coarse else [('^layers\\.(\\d+).downsample', (0,)), ('^layers\\.(\\d+)\\.\\w+\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SwinTransformerV2(nn.Module):
    """ Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(self.patch_embed.grid_size[0] // 2 ** i_layer, self.patch_embed.grid_size[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = {'absolute_pos_embed'}
        for n, m in self.named_modules():
            if any([(kw in n) for kw in ('cpb_mlp', 'logit_scale', 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^absolute_pos_embed|patch_embed', blocks='^layers\\.(\\d+)' if coarse else [('^layers\\.(\\d+).downsample', (0,)), ('^layers\\.(\\d+)\\.\\w+\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def bchw_to_bhwc(x: torch.Tensor) ->torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


class SwinTransformerStage(nn.Module):
    """This class implements a stage of the Swin transformer including multiple layers.
    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(self, embed_dim: int, depth: int, downscale: bool, num_heads: int, feat_size: Tuple[int, int], window_size: Tuple[int, int], mlp_ratio: float=4.0, init_values: Optional[float]=0.0, drop: float=0.0, drop_attn: float=0.0, drop_path: Union[List[float], float]=0.0, norm_layer: Type[nn.Module]=nn.LayerNorm, extra_norm_period: int=0, extra_norm_stage: bool=False, sequential_attn: bool=False) ->None:
        super(SwinTransformerStage, self).__init__()
        self.downscale: bool = downscale
        self.grad_checkpointing: bool = False
        self.feat_size: Tuple[int, int] = (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size
        self.downsample = PatchMerging(embed_dim, norm_layer=norm_layer) if downscale else nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False
        embed_dim = embed_dim * 2 if downscale else embed_dim
        self.blocks = nn.Sequential(*[SwinTransformerBlock(dim=embed_dim, num_heads=num_heads, feat_size=self.feat_size, window_size=window_size, shift_size=tuple([(0 if index % 2 == 0 else w // 2) for w in window_size]), mlp_ratio=mlp_ratio, init_values=init_values, drop=drop, drop_attn=drop_attn, drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path, extra_norm=_extra_norm(index), sequential_attn=sequential_attn, norm_layer=norm_layer) for index in range(depth)])

    def update_input_size(self, new_window_size: int, new_feat_size: Tuple[int, int]) ->None:
        """Method updates the resolution to utilize and the window size and so the pair-wise relative positions.
        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = (new_feat_size[0] // 2, new_feat_size[1] // 2) if self.downscale else new_feat_size
        for block in self.blocks:
            block.update_input_size(new_window_size=new_window_size, new_feat_size=self.feat_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        x = self.downsample(x)
        B, C, H, W = x.shape
        L = H * W
        x = bchw_to_bhwc(x).reshape(B, L, C)
        for block in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = bhwc_to_bchw(x.reshape(B, H, W, -1))
        return x


def init_weights(module: nn.Module, name: str=''):
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        elif 'head' in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class SwinTransformerV2Cr(nn.Module):
    """ Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883
    Args:
        img_size (Tuple[int, int]): Input resolution.
        window_size (Optional[int]): Window size. If None, img_size // window_div. Default: None
        img_window_ratio (int): Window size to image size ratio. Default: 32
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input channels.
        depths (int): Depth of the stage (number of layers).
        num_heads (int): Number of attention heads to be utilized.
        embed_dim (int): Patch embedding dimension. Default: 96
        num_classes (int): Number of output classes. Default: 1000
        mlp_ratio (int):  Ratio of the hidden dimension in the FFN to the input channels. Default: 4
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Dropout rate of attention map. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed. Default: False
    """

    def __init__(self, img_size: Tuple[int, int]=(224, 224), patch_size: int=4, window_size: Optional[int]=None, img_window_ratio: int=32, in_chans: int=3, num_classes: int=1000, embed_dim: int=96, depths: Tuple[int, ...]=(2, 2, 6, 2), num_heads: Tuple[int, ...]=(3, 6, 12, 24), mlp_ratio: float=4.0, init_values: Optional[float]=0.0, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer: Type[nn.Module]=nn.LayerNorm, extra_norm_period: int=0, extra_norm_stage: bool=False, sequential_attn: bool=False, global_pool: str='avg', weight_init='skip', **kwargs: Any) ->None:
        super(SwinTransformerV2Cr, self).__init__()
        img_size = to_2tuple(img_size)
        window_size = tuple([(s // img_window_ratio) for s in img_size]) if window_size is None else to_2tuple(window_size)
        self.num_classes: int = num_classes
        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patch_grid_size: Tuple[int, int] = self.patch_embed.grid_size
        drop_path_rate = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        stages = []
        for index, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stage_scale = 2 ** max(index - 1, 0)
            stages.append(SwinTransformerStage(embed_dim=embed_dim * stage_scale, depth=depth, downscale=index != 0, feat_size=(patch_grid_size[0] // stage_scale, patch_grid_size[1] // stage_scale), num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, init_values=init_values, drop=drop_rate, drop_attn=attn_drop_rate, drop_path=drop_path_rate[sum(depths[:index]):sum(depths[:index + 1])], extra_norm_period=extra_norm_period, extra_norm_stage=extra_norm_stage or index + 1 == len(depths), sequential_attn=sequential_attn, norm_layer=norm_layer))
        self.stages = nn.Sequential(*stages)
        self.global_pool: str = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes else nn.Identity()
        if weight_init != 'skip':
            named_apply(init_weights, self)

    def update_input_size(self, new_img_size: Optional[Tuple[int, int]]=None, new_window_size: Optional[int]=None, img_window_ratio: int=32) ->None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.
        Args:
            new_window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            new_img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            img_window_ratio (int): divisor for calculating window size from image size
        """
        if new_img_size is None:
            new_img_size = self.img_size
        else:
            new_img_size = to_2tuple(new_img_size)
        if new_window_size is None:
            new_window_size = tuple([(s // img_window_ratio) for s in new_img_size])
        new_patch_grid_size = new_img_size[0] // self.patch_size, new_img_size[1] // self.patch_size
        for index, stage in enumerate(self.stages):
            stage_scale = 2 ** max(index - 1, 0)
            stage.update_input_size(new_window_size=new_window_size, new_img_size=(new_patch_grid_size[0] // stage_scale, new_patch_grid_size[1] // stage_scale))

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^patch_embed', blocks='^stages\\.(\\d+)' if coarse else [('^stages\\.(\\d+).downsample', (0,)), ('^stages\\.(\\d+)\\.\\w+\\.(\\d+)', None)])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self) ->nn.Module:
        """Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str]=None) ->None:
        """Method results the classification head
        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
        """
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) ->torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(2, 3))
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class CondConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, K=K, init_weight=init_weight)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None
        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = self.bias.view(self.K, -1)
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        output = output.view(bs, self.out_planes, h, w)
        return output


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class DynamicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4, temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature, init_weight=init_weight)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None
        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = self.bias.view(self.K, -1)
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        output = output.view(bs, self.out_planes, h, w)
        return output


class GlobalLocalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        self.pre_norm = LayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-06, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        x2 = x2
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class gnconv(nn.Module):

    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [(dim // 2 ** i) for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x


class Involution(nn.Module):

    def __init__(self, kernel_size, in_channel=4, stride=1, group=1, ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.stride = stride
        self.group = group
        assert self.in_channel % group == 0
        self.group_channel = self.in_channel // group
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel // ratio, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channel // ratio, self.group * self.kernel_size * self.kernel_size, kernel_size=1)
        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.avgpool(inputs)))))
        b, c, h, w = weight.shape
        weight = weight.reshape(b, self.group, self.kernel_size * self.kernel_size, h, w).unsqueeze(2)
        x_unfold = self.unfold(inputs)
        x_unfold = x_unfold.reshape(B, self.group, C // self.group, self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)
        out = (x_unfold * weight).sum(dim=3)
        out = out.reshape(B, C, H // self.stride, W // self.stride)
        return out


class SpatialGatingUnit(nn.Module):

    def __init__(self, dim, len_sen):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(len_sen, len_sen, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        res, gate = torch.chunk(x, 2, -1)
        gate = self.ln(gate)
        gate = self.proj(gate)
        return res * gate


def exist(x):
    return x is not None


class gMLP(nn.Module):

    def __init__(self, num_tokens=None, len_sen=49, dim=512, d_ff=1024, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, dim) if exist(num_tokens) else nn.Identity()
        self.gmlp = nn.ModuleList([Residual(nn.Sequential(OrderedDict([('ln1_%d' % i, nn.LayerNorm(dim)), ('fc1_%d' % i, nn.Linear(dim, d_ff * 2)), ('gelu_%d' % i, nn.GELU()), ('sgu_%d' % i, SpatialGatingUnit(d_ff, len_sen)), ('fc2_%d' % i, nn.Linear(d_ff, dim))]))) for i in range(num_layers)])
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_tokens), nn.Softmax(-1))

    def forward(self, x):
        embeded = self.embedding(x)
        y = nn.Sequential(*self.gmlp)(embeded)
        logits = self.to_logits(y)
        return logits


class MlpBlock(nn.Module):

    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):

    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        y = self.ln(x)
        y = y.transpose(1, 2)
        y = self.tokens_mlp_block(y)
        y = y.transpose(1, 2)
        out = x + y
        y = self.ln(out)
        y = out + self.channels_mlp_block(y)
        return y


class MlpMixer(nn.Module):

    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim, channels_hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.embd = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

    def forward(self, x):
        y = self.embd(x)
        bs, c, h, w = y.shape
        y = y.view(bs, c, -1).transpose(1, 2)
        if self.tokens_mlp_dim != y.shape[1]:
            raise ValueError('Tokens_mlp_dim is not correct.')
        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)
        y = self.ln(y)
        y = torch.mean(y, dim=1, keepdim=False)
        probs = self.fc(y)
        return probs


class RepMLP(nn.Module):

    def __init__(self, C, O, H, W, h, w, fc1_fc2_reduction=1, fc3_groups=8, repconv_kernels=None, deploy=False):
        super().__init__()
        self.C = C
        self.O = O
        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.fc1_fc2_reduction = fc1_fc2_reduction
        self.repconv_kernels = repconv_kernels
        self.h_part = H // h
        self.w_part = W // w
        self.deploy = deploy
        self.fc3_groups = fc3_groups
        assert H % h == 0
        assert W % w == 0
        self.is_global_perceptron = H != h or W != w
        if self.is_global_perceptron:
            if not self.deploy:
                self.avg = nn.Sequential(OrderedDict([('avg', nn.AvgPool2d(kernel_size=(self.h, self.w))), ('bn', nn.BatchNorm2d(num_features=C))]))
            else:
                self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
            hidden_dim = self.C // self.fc1_fc2_reduction
            self.fc1_fc2 = nn.Sequential(OrderedDict([('fc1', nn.Linear(C * self.h_part * self.w_part, hidden_dim)), ('relu', nn.ReLU()), ('fc2', nn.Linear(hidden_dim, C * self.h_part * self.w_part))]))
        self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, kernel_size=1, groups=fc3_groups, bias=self.deploy)
        self.fc3_bn = nn.Identity() if self.deploy else nn.BatchNorm2d(self.O * self.h * self.w)
        if not self.deploy and self.repconv_kernels is not None:
            for k in self.repconv_kernels:
                repconv = nn.Sequential(OrderedDict([('conv', nn.Conv2d(self.C, self.O, kernel_size=k, padding=(k - 1) // 2, groups=fc3_groups, bias=False)), ('bn', nn.BatchNorm2d(self.O))]))
                self.__setattr__('repconv{}'.format(k), repconv)

    def switch_to_deploy(self):
        self.deploy = True
        fc1_weight, fc1_bias, fc3_weight, fc3_bias = self.get_equivalent_fc1_fc3_params()
        if self.repconv_kernels is not None:
            for k in self.repconv_kernels:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, 1, 1, 0, bias=True, groups=self.fc3_groups)
        self.fc3_bn = nn.Identity()
        if self.is_global_perceptron:
            self.__delattr__('avg')
            self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.data = fc1_weight
            self.fc1_fc2.fc1.bias.data = fc1_bias
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def get_equivalent_fc1_fc3_params(self):
        fc_weight, fc_bias = self._fuse_bn(self.fc3, self.fc3_bn)
        if self.repconv_kernels is not None:
            max_kernel = max(self.repconv_kernels)
            max_branch = self.__getattr__('repconv{}'.format(max_kernel))
            conv_weight, conv_bias = self._fuse_bn(max_branch.conv, max_branch.bn)
            for k in self.repconv_kernels:
                if k != max_kernel:
                    tmp_branch = self.__getattr__('repconv{}'.format(k))
                    tmp_weight, tmp_bias = self._fuse_bn(tmp_branch.conv, tmp_branch.bn)
                    tmp_weight = F.pad(tmp_weight, [(max_kernel - k) // 2] * 4)
                    conv_weight += tmp_weight
                    conv_bias += tmp_bias
            repconv_weight, repconv_bias = self._conv_to_fc(conv_weight, conv_bias)
            final_fc3_weight = fc_weight + repconv_weight.reshape_as(fc_weight)
            final_fc3_bias = fc_bias + repconv_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        if self.is_global_perceptron:
            avgbn = self.avg.bn
            std = (avgbn.running_var + avgbn.eps).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn.running_mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.in_features // len(avgbias)
            replicated_avgbias = avgbias.repeat_interleave(replicate_times).view(-1, 1)
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            final_fc1_bias = fc1.bias + bias_diff
            final_fc1_weight = fc1.weight * scale.repeat_interleave(replicate_times).view(1, -1)
        else:
            final_fc1_weight = None
            final_fc1_bias = None
        return final_fc1_weight, final_fc1_bias, final_fc3_weight, final_fc3_bias

    def _conv_to_fc(self, conv_kernel, conv_bias):
        I = torch.eye(self.C * self.h * self.w // self.fc3_groups).repeat(1, self.fc3_groups).reshape(self.C * self.h * self.w // self.fc3_groups, self.C, self.h, self.w)
        fc_k = F.conv2d(I, conv_kernel, padding=conv_kernel.size(2) // 2, groups=self.fc3_groups)
        fc_k = fc_k.reshape(self.C * self.h * self.w // self.fc3_groups, self.O * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias

    def _fuse_bn(self, conv_or_fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        if conv_or_fc.weight.ndim == 4:
            t = t.reshape(-1, 1, 1, 1)
        else:
            t = t.reshape(-1, 1)
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def forward(self, x):
        if self.is_global_perceptron:
            input = x
            v = self.avg(x)
            v = v.reshape(-1, self.C * self.h_part * self.w_part)
            v = self.fc1_fc2(v)
            v = v.reshape(-1, self.C, self.h_part, 1, self.w_part, 1)
            input = input.reshape(-1, self.C, self.h_part, self.h, self.w_part, self.w)
            input = v + input
        else:
            input = x.view(-1, self.C, self.h_part, self.h, self.w_part, self.w)
        partition = input.permute(0, 2, 4, 1, 3, 5)
        fc3_out = partition.reshape(-1, self.C * self.h * self.w, 1, 1)
        fc3_out = self.fc3_bn(self.fc3(fc3_out))
        fc3_out = fc3_out.reshape(-1, self.h_part, self.w_part, self.O, self.h, self.w)
        if self.repconv_kernels is not None and not self.deploy:
            conv_input = partition.reshape(-1, self.C, self.h, self.w)
            conv_out = 0
            for k in self.repconv_kernels:
                repconv = self.__getattr__('repconv{}'.format(k))
                conv_out += repconv(conv_input)
            conv_out = conv_out.view(-1, self.h_part, self.w_part, self.O, self.h, self.w)
            fc3_out += conv_out
        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)
        fc3_out = fc3_out.reshape(-1, self.C, self.H, self.W)
        return fc3_out


class Rearange(nn.Module):

    def __init__(self, image_size=14, patch_size=7):
        self.h = patch_size
        self.w = patch_size
        self.nw = image_size // patch_size
        self.nh = image_size // patch_size
        num_patches = (image_size // patch_size) ** 2
        super().__init__()

    def forward(self, x):
        bs, c, H, W = x.shape
        y = x.reshape(bs, c, self.h, self.nh, self.w, self.nw)
        y = y.permute(0, 3, 5, 2, 4, 1)
        y = y.contiguous().view(bs, self.nh * self.nw, -1)
        return y


class Affine(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, channel))
        self.b = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):

    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-05
        else:
            init_eps = 1e-06
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Module):

    def __init__(self, dim=128, image_size=14, patch_size=7, expansion_factor=4, depth=4, class_num=1000):
        super().__init__()
        self.flatten = Rearange(image_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)
        self.embedding = nn.Linear(patch_size ** 2 * 3, dim)
        self.mlp = nn.Sequential()
        for i in range(depth):
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Conv1d(patch_size ** 2, patch_size ** 2, 1)))
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Sequential(nn.Linear(dim, dim * expansion_factor), nn.GELU(), nn.Linear(dim * expansion_factor, dim))))
        self.aff = Affine(dim)
        self.classifier = nn.Linear(dim, class_num)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        y = self.flatten(x)
        y = self.embedding(y)
        y = self.mlp(y)
        y = self.aff(y)
        y = torch.mean(y, dim=1)
        out = self.softmax(self.classifier(y))
        return out


class sMLPBlock(nn.Module):

    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3 * c, c)

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3.0, qkv_bias=False, qk_scale=None, attn_drop=0, drop_path_rate=0.0, skip_lam=1.0, mlp_fn=WeightedPermuteMLP, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)
    return blocks


class VisionPermutator(nn.Module):
    """ Vision Permutator
    """

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))


def _conv_bn(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm2d(output_channel))
    return res


class ACNet(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size=3, groups=1, stride=1, deploy=False, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.groups = groups
        self.activation = nn.ReLU()
        if not self.deploy:
            self.brb_3x3 = _conv_bn(input_channel, output_channel, kernel_size=3, padding=1, groups=groups)
            self.brb_1x3 = _conv_bn(input_channel, output_channel, kernel_size=(1, 3), padding=(0, 1), groups=groups)
            self.brb_3x1 = _conv_bn(input_channel, output_channel, kernel_size=(3, 1), padding=(1, 0), groups=groups)
        else:
            self.brb_rep = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode='zeros', stride=stride, bias=True)

    def forward(self, inputs):
        if self.deploy:
            return self.activation(self.brb_rep(inputs))
        return self.activation(self.brb_1x3(inputs) + self.brb_3x1(inputs) + self.brb_3x3(inputs))

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.conv.in_channels, out_channels=self.brb_3x3.conv.out_channels, kernel_size=self.brb_3x3.conv.kernel_size, padding=self.brb_3x3.conv.padding, padding_mode=self.brb_3x3.conv.padding_mode, stride=self.brb_3x3.conv.stride, groups=self.brb_3x3.conv.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_3x1')
        self.__delattr__('brb_1x3')

    def _pad_1x3_kernel(self, kernel):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [0, 0, 1, 1])

    def _pad_3x1_kernel(self, kernel):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [1, 1, 0, 0])

    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight, brb_3x3_bias = self._fuse_conv_bn(self.brb_3x3)
        brb_1x3_weight, brb_1x3_bias = self._fuse_conv_bn(self.brb_1x3)
        brb_3x1_weight, brb_3x1_bias = self._fuse_conv_bn(self.brb_3x1)
        return brb_3x3_weight + self._pad_1x3_kernel(brb_1x3_weight) + self._pad_3x1_kernel(brb_3x1_weight), brb_3x3_bias + brb_1x3_bias + brb_3x1_bias

    def _fuse_conv_bn(self, branch):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepBlock(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size=3, groups=1, stride=1, deploy=False, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.groups = groups
        self.activation = nn.ReLU()
        assert self.kernel_size == 3
        assert self.padding == 1
        if not self.deploy:
            self.brb_3x3 = _conv_bn(input_channel, output_channel, kernel_size=self.kernel_size, padding=self.padding, groups=groups)
            self.brb_1x1 = _conv_bn(input_channel, output_channel, kernel_size=1, padding=0, groups=groups)
            self.brb_identity = nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None
        else:
            self.brb_rep = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode='zeros', stride=stride, bias=True)

    def forward(self, inputs):
        if self.deploy:
            return self.activation(self.brb_rep(inputs))
        if self.brb_identity == None:
            identity_out = 0
        else:
            identity_out = self.brb_identity(inputs)
        return self.activation(self.brb_1x1(inputs) + self.brb_3x3(inputs) + identity_out)

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.conv.in_channels, out_channels=self.brb_3x3.conv.out_channels, kernel_size=self.brb_3x3.conv.kernel_size, padding=self.brb_3x3.conv.padding, padding_mode=self.brb_3x3.conv.padding_mode, stride=self.brb_3x3.conv.stride, groups=self.brb_3x3.conv.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_1x1')
        self.__delattr__('brb_identity')

    def _pad_1x1_kernel(self, kernel):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [1] * 4)

    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight, brb_3x3_bias = self._fuse_conv_bn(self.brb_3x3)
        brb_1x1_weight, brb_1x1_bias = self._fuse_conv_bn(self.brb_1x1)
        brb_id_weight, brb_id_bias = self._fuse_conv_bn(self.brb_identity)
        return brb_3x3_weight + self._pad_1x1_kernel(brb_1x1_weight) + brb_id_weight, brb_3x3_bias + brb_1x1_bias + brb_id_bias

    def _fuse_conv_bn(self, branch):
        if branch is None:
            return 0, 0
        elif isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.input_channel // self.groups
                kernel_value = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ACNet,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ACmix,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Affine,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttentionGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AxialPositionalEmbedding,
     lambda: ([], {'dim': 4, 'shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBAMBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (CBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChanLayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 1, 49])], {}),
     False),
    (ClassBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CoAtNet,
     lambda: ([], {'in_ch': 4, 'image_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (CoTAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (ConvBN,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordAtt,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CrossAttentionBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DAModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 1, 49])], {}),
     False),
    (Depth_Pointwise_Conv1d,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DepthwiseSeparableConvolution,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Deterministic,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DoubleAttention,
     lambda: ([], {'in_channels': 4, 'c_m': 4, 'c_n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'in_embed_dim': 4, 'out_embed_dim': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ECAAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExternalAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'dim': 4, 'mlp_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalLocalFilter,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Image2Tokens,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale_Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LayerScale_Block_CA,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Layer_scale_init_Block_only_token,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Learned_Aggregation_Layer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConvBlock,
     lambda: ([], {'ksize': 4, 'input_filters': 4, 'output_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'planes': 4, 'mlp_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MUSEAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MV2Block,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Meta4D,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MlpBlock,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileViTv2Attention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutlookAttention,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     False),
    (ParNetAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (ParallelPolarizedSelfAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PatchEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PermutatorBlock,
     lambda: ([], {'dim': 4, 'segment_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     False),
    (PreAffinePostLayerScale,
     lambda: ([], {'dim': 4, 'depth': 1, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RepBlock,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RepMLP,
     lambda: ([], {'C': 4, 'O': 4, 'H': 4, 'W': 4, 'h': 4, 'w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (SE,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SKAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'dim': 4, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SequentialPolarizedSelfAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (SimAM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimplifiedScaledDotProductAttention,
     lambda: ([], {'d_model': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1])], {}),
     False),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGroupEnhance,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerMLPWithConv,
     lambda: ([], {'channels': 4, 'expansion': 4, 'drop': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TripletAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UFOAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (ZPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_embedding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'patch_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_head_pooling,
     lambda: ([], {'in_feature': 4, 'out_feature': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xmu_xiaoma666_External_Attention_pytorch(_paritybench_base):
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

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

    def test_077(self):
        self._check(*TESTCASES[77])

    def test_078(self):
        self._check(*TESTCASES[78])

    def test_079(self):
        self._check(*TESTCASES[79])

    def test_080(self):
        self._check(*TESTCASES[80])

