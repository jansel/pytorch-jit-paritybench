import sys
_module = sys.modules[__name__]
del sys
models = _module
alexnet = _module
alexnet_dnn_block = _module
alexnet_mcdo_block = _module
alternet = _module
attentions = _module
cbamresnet = _module
cbamresnet_dnn_block = _module
cbamresnet_mcdo_block = _module
classifier_block = _module
convit = _module
embeddings = _module
ensemble = _module
gates = _module
layers = _module
mixer = _module
mobilenet = _module
pit = _module
preresnet = _module
preresnet_dnn_block = _module
preresnet_mcdo_block = _module
prevggnet = _module
prevggnet_dnn_block = _module
prevggnet_mcdo_block = _module
resnet = _module
resnet_dnn_block = _module
resnet_mcdo_block = _module
resnext = _module
seresnet = _module
seresnet_dnn_block = _module
seresnet_mcdo_block = _module
smoothing_block = _module
swin = _module
vggnet = _module
vggnet_dnn_block = _module
vggnet_mcdo_block = _module
vit = _module
wideresnet = _module
ops = _module
adversarial = _module
arithmetic = _module
cifarc = _module
cifarp = _module
datasets = _module
imagenetc = _module
loss_landscapes = _module
meters = _module
norm = _module
schedulers = _module
tests = _module
trains = _module

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


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from functools import partial


from itertools import cycle


from torch import nn


from torch import einsum


import types


import math


import random


from math import sqrt


from typing import Any


from typing import Callable


from typing import Optional


from typing import Tuple


import torchvision.transforms as transforms


from torchvision.datasets.vision import VisionDataset


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import download_and_extract_archive


import torchvision


import torchvision.datasets as datasets


import copy


import re


from torch.optim.lr_scheduler import _LRScheduler


import matplotlib.pyplot as plt


import torch.optim as optim


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, rate=0.3, **block_kwargs):
        super(BasicBlock, self).__init__()
        self.rate = rate
        self.conv = layers.conv3x3(in_channels, out_channels)
        self.bn = layers.bn(out_channels)
        self.relu = layers.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.rate)
        return x

    def extra_repr(self):
        return 'rate=%.3e' % self.rate


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out
        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim_out, 1), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = self.to_q(x), *self.to_kv(x).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)
        out = self.to_out(out)
        return out, attn


class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, window_size=7, k=1, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size
        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p
        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        x = rearrange(x, 'b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2', p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, '(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)', n1=n1, n2=n2, p1=p, p2=p)
        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]
        return d


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x

    def extra_repr(self):
        return 'p=%s' % repr(self.p)


def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)


class AttentionBlockA(nn.Module):
    expansion = 4

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, sd=0.0, stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU, **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion
        self.shortcut = []
        if dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion))
            self.shortcut.append(norm(dim_out * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.conv = nn.Sequential(conv1x1(dim_in, width, stride=stride), norm(width), activation())
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = norm(dim_out * self.expansion)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv(x)
        x, attn = self.attn(x)
        x = self.norm(x)
        x = self.sd(x) + skip
        return x


class AttentionBasicBlockA(AttentionBlockA):
    expansion = 1


class AttentionBlockB(nn.Module):
    expansion = 4

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, sd=0.0, stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU, **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion
        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(layers.conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()
        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)
        x = self.sd(x) + skip
        return x


class AttentionBasicBlockB(AttentionBlockB):
    expansion = 1


class StemA(nn.Module):

    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()
        self.layer0 = []
        if pool:
            self.layer0.append(layers.convnxn(dim_in, dim_out, kernel_size=7, stride=2, padding=3))
            self.layer0.append(layers.bn(dim_out))
            self.layer0.append(layers.relu())
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(layers.conv3x3(dim_in, dim_out, stride=1))
            self.layer0.append(layers.bn(dim_out))
            self.layer0.append(layers.relu())
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


class StemB(nn.Module):

    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()
        self.layer0 = []
        if pool:
            self.layer0.append(layers.convnxn(dim_in, dim_out, kernel_size=7, stride=2, padding=3))
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(layers.conv3x3(dim_in, dim_out, stride=1))
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(f(dim_in, hidden_dim), activation(), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(), f(hidden_dim, dim_out), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim_out), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0, attn=Attention1d, norm=nn.LayerNorm, f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()
        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip
        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0, reduction=16, **block_kwargs):
        super(Bottleneck, self).__init__()
        width = int(channels * (width_per_group / 64.0)) * groups
        self.rate = rate
        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.conv1 = nn.Sequential(layers.conv1x1(in_channels, width), layers.bn(width), layers.relu())
        self.conv2 = nn.Sequential(layers.conv3x3(width, width, stride=stride, groups=groups), layers.bn(width), layers.relu())
        self.conv3 = nn.Sequential(layers.conv1x1(width, channels * self.expansion), layers.bn(channels * self.expansion))
        self.relu = layers.relu()
        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()
        self.gate = gates.ChannelGate(channels * self.expansion, reduction, max_pool=False)

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv3(x)
        x = self.gate(x)
        x = self.sd(x) + skip
        x = self.relu(x)
        return x

    def extra_repr(self):
        return 'rate=%.3e' % self.rate


class GAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class BNGAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(BNGAPBlock, self).__init__()
        self.bn = layers.bn(in_features)
        self.relu = layers.relu()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class MLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()
        self.dense1 = layers.dense(in_features, 4096)
        self.relu1 = layers.relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = layers.dense(4096, 4096)
        self.relu2 = layers.relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = layers.dense(4096, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x


class GMaxPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMaxPBlock, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gmp(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class GMedPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMedPBlock, self).__init__()
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        x = torch.topk(x, k=int(x.size()[2] / 2), dim=2)[0][:, :, -1]
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class GAPClipBlock(nn.Module):

    def __init__(self, in_features, num_classes, temp=2.0, **kwargs):
        super(GAPClipBlock, self).__init__()
        self.temp = temp
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = self.temp * (F.sigmoid(x / self.temp) - 0.5)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class GAPMLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPMLPBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = layers.dense(in_features, 4096)
        self.relu1 = layers.relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = layers.dense(4096, 4096)
        self.relu2 = layers.relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = layers.dense(4096, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x


class ConvAttention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, k=1, kernel_size=1, dilation=1, padding=0, stride=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.conv_args = {'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride}
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out
        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim_out, 1), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        q, kv = self.to_q(x), self.to_kv(x).chunk(2, dim=1)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        q = repeat(q, 'b h n d -> b h n w d', w=self.conv_args['kernel_size'] ** 2)
        k, v = map(lambda t: F.unfold(t, **self.conv_args), kv)
        k, v = map(lambda t: rearrange(t, 'b (h d w) n -> b h n w d', h=self.heads, d=q.shape[-1]), (k, v))
        dots = einsum('b h n w d, b h n w d -> b h n w', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        out = einsum('b h n w, b h n w d -> b h n d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)
        out = self.to_out(out)
        return out, attn

    def extra_repr(self):
        return ', '.join([('%s=%s' % (k, v)) for k, v in self.conv_args.items()])


class AbsPosEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim, stride=None, cls=True):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception('Image dimensions must be divisible by the patch size.')
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(image_size, patch_size, stride)
        num_patches = output_size ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embedding
        return x

    @staticmethod
    def _conv_output_size(image_size, kernel_size, stride, padding=0):
        return int((image_size - kernel_size + 2 * padding) / stride + 1)


class ConViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, depth, dim, heads, dim_mlp, channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, sd=0.0, k=1, kernel_size=3, dilation=1, padding=0, stride=1, embedding=None, classifier=None, name='convit', **block_kwargs):
        super().__init__()
        self.name = name
        self.embedding = nn.Sequential(nn.Conv2d(channel, dim, patch_size, stride=patch_size, padding=0), Rearrange('b c x y -> b (x y) c'), AbsPosEmbedding(image_size, patch_size, dim, cls=False), nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(), Rearrange('b (x y) c -> b c x y', y=image_size // patch_size)) if embedding is None else embedding
        attn = partial(ConvAttention2d, k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        f = partial(nn.Conv2d, kernel_size=1, stride=1)
        self.transformers = []
        for i in range(depth):
            self.transformers.append(Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp, attn=attn, f=f, norm=layers.ln2d, dropout=dropout, sd=sd * i / (depth - 1)))
        self.transformers = nn.Sequential(*self.transformers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(dim), nn.Linear(dim, num_classes)) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)
        return x


class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim_out, channel=3):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception('Image dimensions must be divisible by the patch size.')
        patch_dim = channel * patch_size ** 2
        self.patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim_out))

    def forward(self, x):
        x = self.patch_embedding(x)
        return x


class CLSToken(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class PatchUnembedding(nn.Module):

    def __init__(self, image_size, patch_size):
        super().__init__()
        h, w = image_size // patch_size, image_size // patch_size
        self.rearrange = nn.Sequential(Rearrange('b (h w) (p1 p2 d) -> b d (h p1) (w p2)', h=h, w=w, p1=patch_size, p2=patch_size))

    def forward(self, x):
        x = x[:, 1:]
        x = self.rearrange(x)
        return x


class ConvEmbedding(nn.Module):

    def __init__(self, patch_size, dim_out, channel=3, stride=None):
        super().__init__()
        stride = patch_size if stride is None else stride
        patch_dim = channel * patch_size ** 2
        self.patch_embedding = nn.Sequential(nn.Unfold(kernel_size=patch_size, stride=stride), Rearrange('b c n -> b n c'), nn.Linear(patch_dim, dim_out))

    def forward(self, x):
        x = self.patch_embedding(x)
        return x


class Ensemble(nn.Module):

    def __init__(self, models, name=None):
        super(Ensemble, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = '%s_ensemble' % models[0].name
        self.models = nn.ModuleList(models)

    def forward(self, x):
        xs = torch.stack([model(x) for model in self.models])
        xs = xs - torch.logsumexp(xs, dim=-1, keepdim=True)
        x = torch.logsumexp(xs, dim=0)
        return x


class ChannelGate(nn.Module):

    def __init__(self, channel, reduction=16, max_pool=True):
        super().__init__()
        self.pools = []
        self.pools.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.pools.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.pools = self.pools if max_pool else self.pools[:1]
        self.ff = nn.Sequential(layers.dense(channel, channel // reduction, bias=False), layers.relu(), layers.dense(channel // reduction, channel, bias=False))
        self.prob = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        s = torch.cat([pool(x) for pool in self.pools], dim=-1)
        s = rearrange(s, 'b c n m -> b (n m) c')
        s = self.ff(s)
        s = reduce(s, 'b n c -> b c', 'mean')
        s = self.prob(s)
        s = s.view(b, c, 1, 1)
        return x * s


class SpatialGate(nn.Module):

    def __init__(self, kernel_size=7, max_pool=True):
        super().__init__()
        self.pools = []
        self.pools.append(partial(torch.mean, dim=1, keepdim=True))
        self.pools.append(lambda x: partial(torch.max, dim=1, keepdim=True)(x)[0])
        self.pools = self.pools if max_pool else self.pools[:1]
        self.ff = nn.Sequential(layers.convnxn(len(self.pools), 1, kernel_size=7, stride=1, padding=(kernel_size - 1) // 2), layers.bn(1))
        self.prob = nn.Sigmoid()

    def forward(self, x):
        s = torch.cat([pool(x) for pool in self.pools], dim=1)
        s = self.ff(s)
        s = self.prob(s)
        return x * s


class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode='constant', **kwargs):
        super(SamePad, self).__init__()
        self.pad_size = [int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)), int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0))]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)
        return x

    def extra_repr(self):
        return 'pad_size=%s, pad_mode=%s' % (self.pad_size, self.pad_mode)


class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode='replicate', **kwargs):
        super(Blur, self).__init__()
        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)
        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])
        return x

    def extra_repr(self):
        return 'pad=%s, filter_proto=%s' % (self.pad, self.filter_proto.tolist())


class Downsample(nn.Module):

    def __init__(self, strides=(2, 2), **kwargs):
        super(Downsample, self).__init__()
        if isinstance(strides, int):
            strides = strides, strides
        self.strides = strides

    def forward(self, x):
        shape = -(-x.size()[2] // self.strides[0]), -(-x.size()[3] // self.strides[1])
        x = F.interpolate(x, size=shape, mode='nearest')
        return x

    def extra_repr(self):
        return 'strides=%s' % repr(self.strides)


class Lambda(nn.Module):

    def __init__(self, lmd):
        super().__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception('`lmd` should be lambda ftn.')
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)


class MixerBlock(nn.Module):

    def __init__(self, hidden_dim, spatial_dim, channel_dim, num_patches, dropout=0.0, sd=0.0):
        super().__init__()
        f1, f2 = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff1 = FeedForward(num_patches, spatial_dim, f=f1, dropout=dropout)
        self.sd1 = DropPath(sd)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff2 = FeedForward(hidden_dim, channel_dim, f=f2, dropout=dropout)
        self.sd2 = DropPath(sd)

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.sd1(x) + skip
        skip = x
        x = self.norm2(x)
        x = self.ff2(x)
        x = self.sd2(x) + skip
        return x


class Mixer(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, hidden_dim, spatial_dim, channel_dim, depth, channel=3, dropout=0.0, sd=0.0, embedding=None, classifier=None, name='mixer'):
        super().__init__()
        self.name = name
        if image_size % patch_size != 0:
            raise Exception('Image must be divisible by patch size.')
        num_patches = (image_size // patch_size) ** 2
        self.embedding = nn.Sequential(PatchEmbedding(image_size, patch_size, hidden_dim, channel=channel)) if embedding is None else embedding
        self.mlps = []
        for i in range(depth):
            self.mlps.append(MixerBlock(hidden_dim, spatial_dim, channel_dim, num_patches, dropout=dropout, sd=sd * i / (depth - 1)))
        self.mlps = nn.Sequential(*self.mlps)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), Reduce('b n c -> b c', 'mean'), nn.Linear(hidden_dim, num_classes)) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.mlps(x)
        x = self.classifier(x)
        return x


class Basic(nn.Module):

    def __init__(self, dim_in, dim_out, stride, expand_ratio, **block_kwargs):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(dim_in * expand_ratio)
        self.identity = stride == 1 and dim_in == dim_out
        self.conv1 = nn.Sequential(layers.conv3x3(dim_in, hidden_dim, stride=stride, groups=dim_in), layers.bn(hidden_dim), layers.relu6())
        self.conv2 = nn.Sequential(layers.conv1x1(hidden_dim, dim_out), layers.bn(dim_out))

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip if self.identity else x
        return x


class DepthwiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias))

    def forward(self, x):
        return self.net(x)


class Pool(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_ff = nn.Linear(dim_in, dim_out)
        self.downsample = DepthwiseConv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        cls_token, spat_tokens = x[:, :1], x[:, 1:]
        _, s, _ = spat_tokens.shape
        h, w = int(sqrt(s)), int(sqrt(s))
        cls_token = self.cls_ff(cls_token)
        spat_tokens = rearrange(spat_tokens, 'b (h w) c -> b c h w', h=h, w=w)
        spat_tokens = self.downsample(spat_tokens)
        spat_tokens = rearrange(spat_tokens, 'b c h w -> b (h w) c')
        return torch.cat((cls_token, spat_tokens), dim=1)


class PiT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dims, depths, heads, dims_head, dims_mlp, channel=3, dropout=0.0, emb_dropout=0.0, sd=0.0, stride=None, embedding=None, classifier=None, name='pit'):
        super().__init__()
        self.name = name
        if len(depths) != 3:
            msg = '`depths` must be a tuple of integers with len of 3, ' + 'specifying the number of blocks before each downsizing.'
            raise Exception(msg)
        dims = self._to_tuple(dims, len(depths))
        dims = dims[0], *dims
        heads = self._to_tuple(heads, len(depths))
        dims_head = self._to_tuple(dims_head, len(depths))
        dims_mlp = self._to_tuple(dims_mlp, len(depths))
        idxs = [[j for j in range(sum(depths[:i]), sum(depths[:i + 1]))] for i in range(len(depths))]
        sds = [[(sd * j / (sum(depths) - 1)) for j in js] for js in idxs]
        pools = [False] + [True] * (len(depths) - 1)
        self.embedding = nn.Sequential(ConvEmbedding(patch_size, dims[0], channel=channel, stride=stride), CLSToken(dims[0]), AbsPosEmbedding(image_size, patch_size, dims[0], stride=stride), nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()) if embedding is None else embedding
        self.transformers = []
        for i in range(len(depths)):
            if pools[i]:
                self.transformers.append(Pool(dims[i], dims[i + 1]))
            for j in range(depths[i]):
                self.transformers.append(Transformer(dims[i + 1], heads=heads[i], dim_head=dims_head[i], dim_mlp=dims_mlp[i], dropout=dropout, sd=sds[i][j]))
        self.transformers = nn.Sequential(*self.transformers)
        self.classifier = nn.Sequential(Lambda(lambda x: x[:, 0]), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes)) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _to_tuple(v, l):
        return v if isinstance(v, tuple) or isinstance(v, list) else (v,) * l


class TanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=10.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(TanhBlurBlock, self).__init__()
        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class BNTanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=10.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(BNTanhBlurBlock, self).__init__()
        self.bn = layers.bn(in_filters)
        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.bn(x)
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class TanhBlock(nn.Module):

    def __init__(self, temp=10.0, **kwargs):
        super(TanhBlock, self).__init__()
        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class SigmoidBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=10.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(SigmoidBlurBlock, self).__init__()
        self.temp = temp
        self.relu = layers.relu()
        self.sigmoid = nn.Sigmoid()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = 4 * self.temp * (self.sigmoid(x / self.temp) - 0.5)
        x = self.relu(x)
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class SoftmaxBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=10.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(SoftmaxBlurBlock, self).__init__()
        self.temp = temp
        self.relu = layers.relu()
        self.softmax = nn.Softmax(dim=1)
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.softmax(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class ReLuBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=6.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(ReLuBlurBlock, self).__init__()
        self.temp = temp
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.temp)
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'thr=%.3e' % self.thr


class ScalingBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=5.0, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(ScalingBlurBlock, self).__init__()
        self.temp = temp
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = x / self.temp
        x = self.blur(x)
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class ScalingBlock(nn.Module):

    def __init__(self, temp=5.0, **kwargs):
        super(ScalingBlock, self).__init__()
        self.temp = temp

    def forward(self, x):
        x = x / self.temp
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


class ReLuBlock(nn.Module):

    def __init__(self, thr=6.0, **kwargs):
        super(ReLuBlock, self).__init__()
        self.thr = thr

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.thr)
        return x

    def extra_repr(self):
        return 'thr=%.3e' % self.thr


class BNBlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(PreactBlurBlock, self).__init__()
        self.bn = layers.bn(in_filters)
        self.relu = layers.relu()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.blur(x)
        return x


class BlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode='constant', **kwargs):
        super(BlurBlock, self).__init__()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.blur(x)
        return x


class CyclicShift(nn.Module):

    def __init__(self, d, dims=(2, 3)):
        super().__init__()
        self.d = d
        self.dims = dims

    def forward(self, x):
        x = torch.roll(x, shifts=(self.d, self.d), dims=self.dims)
        return x


class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, pool):
        super().__init__()
        self.patch_merge = nn.Conv2d(in_channels, out_channels, kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.patch_merge(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=32, dropout=0.0, window_size=7, shifted=False):
        super().__init__()
        self.attn = Attention1d(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.window_size = window_size
        self.shifted = shifted
        self.d = window_size // 2
        self.shift = CyclicShift(-1 * self.d) if shifted else nn.Identity()
        self.backshift = CyclicShift(self.d) if shifted else nn.Identity()
        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p
        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        if self.shifted:
            mask = mask + self._upper_lower_mask(h // p, w // p, p, p, self.d, x.device)
            mask = mask + self._left_right_mask(h // p, w // p, p, p, self.d, x.device)
            mask = repeat(mask, 'n h i j -> (b n) h i j', b=b)
        x = self.shift(x)
        x = rearrange(x, 'b c (n1 p1) (n2 p2) -> (b n1 n2) (p1 p2) c', p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, '(b n1 n2) (p1 p2) c -> b c (n1 p1) (n2 p2)', n1=n1, n2=n2, p1=p, p2=p)
        x = self.backshift(x)
        return x, attn

    @staticmethod
    def _upper_lower_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m[-d * i:, :-d * j] = float('-inf')
        m[:-d * i, -d * j:] = float('-inf')
        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n2:] = mask[-n2:] + m
        return mask

    @staticmethod
    def _left_right_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m = rearrange(m, '(i k) (j l) -> i k j l', i=i, j=j)
        m[:, -d:, :, :-d] = float('-inf')
        m[:, :-d, :, -d:] = float('-inf')
        m = rearrange(m, 'i k j l -> (i k) (j l)')
        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n1 - 1::n1] += m
        return mask

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]
        return d


def ln2d(dim):
    return nn.Sequential(Rearrange('b c h w -> b h w c'), nn.LayerNorm(dim), Rearrange('b h w c -> b c h w'))


class Swin(nn.Module):

    def __init__(self, *, num_classes, depths, dims, heads, dims_mlp, channel=3, dim_head=32, window_size=7, pools=(4, 2, 2, 2), dropout=0.0, sd=0.0, classifier=None, name='swin', **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(depths[:i]), sum(depths[:i + 1]))] for i in range(len(depths))]
        sds = [[(sd * j / (sum(depths) - 1)) for j in js] for js in idxs]
        self.layer1 = self._make_layer(in_channels=channel, hidden_dimension=dims[0], depth=depths[0], window_size=window_size, pool=pools[0], num_heads=heads[0], dim_head=dim_head, dim_mlp=dims_mlp[0], dropout=dropout, sds=sds[0])
        self.layer2 = self._make_layer(in_channels=dims[0], hidden_dimension=dims[1], depth=depths[1], window_size=window_size, pool=pools[1], num_heads=heads[1], dim_head=dim_head, dim_mlp=dims_mlp[1], dropout=dropout, sds=sds[1])
        self.layer3 = self._make_layer(in_channels=dims[1], hidden_dimension=dims[2], depth=depths[2], window_size=window_size, pool=pools[2], num_heads=heads[2], dim_head=dim_head, dim_mlp=dims_mlp[2], dropout=dropout, sds=sds[2])
        self.layer4 = self._make_layer(in_channels=dims[2], hidden_dimension=dims[3], depth=depths[3], window_size=window_size, pool=pools[3], num_heads=heads[3], dim_head=dim_head, dim_mlp=dims_mlp[3], dropout=dropout, sds=sds[3])
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], num_classes)) if classifier is None else classifier

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layer(in_channels, hidden_dimension, depth, window_size, pool, num_heads, dim_head, dim_mlp, dropout, sds):
        attn1 = partial(WindowAttention, window_size=window_size, shifted=False)
        attn2 = partial(WindowAttention, window_size=window_size, shifted=True)
        seq = list()
        seq.append(PatchMerging(in_channels, hidden_dimension, pool))
        for i in range(depth // 2):
            wt = Transformer(hidden_dimension, heads=num_heads, dim_head=dim_head, dim_mlp=dim_mlp, norm=ln2d, attn=attn1, f=partial(nn.Conv2d, kernel_size=1), dropout=dropout, sd=sds[2 * i])
            swt = Transformer(hidden_dimension, heads=num_heads, dim_head=dim_head, dim_mlp=dim_mlp, norm=ln2d, attn=attn2, f=partial(nn.Conv2d, kernel_size=1), dropout=dropout, sd=sds[2 * i + 1])
            seq.extend([wt, swt])
        return nn.Sequential(*seq)


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, depth, dim, heads, dim_mlp, channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, sd=0.0, embedding=None, classifier=None, name='vit', **block_kwargs):
        super().__init__()
        self.name = name
        self.embedding = nn.Sequential(PatchEmbedding(image_size, patch_size, dim, channel=channel), CLSToken(dim), AbsPosEmbedding(image_size, patch_size, dim, cls=True), nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()) if embedding is None else embedding
        self.transformers = []
        for i in range(depth):
            self.transformers.append(Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp, dropout=dropout, sd=sd * i / (depth - 1)))
        self.transformers = nn.Sequential(*self.transformers)
        self.classifier = nn.Sequential(Lambda(lambda x: x[:, 0]), nn.LayerNorm(dim), nn.Linear(dim, num_classes)) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AbsPosEmbedding,
     lambda: ([], {'image_size': 4, 'patch_size': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 2, 4])], {}),
     True),
    (Blur,
     lambda: ([], {'in_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CyclicShift,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim_in': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixerBlock,
     lambda: ([], {'hidden_dim': 4, 'spatial_dim': 4, 'channel_dim': 4, 'num_patches': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PatchMerging,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'pool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReLuBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SamePad,
     lambda: ([], {'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalingBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StemB,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xxxnell_how_do_vits_work(_paritybench_base):
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

