import sys
_module = sys.modules[__name__]
del sys
model_zoo_tables = _module
test_models = _module
pycls = _module
core = _module
benchmark = _module
builders = _module
checkpoint = _module
config = _module
distributed = _module
io = _module
logging = _module
meters = _module
net = _module
optimizer = _module
plotting = _module
timer = _module
trainer = _module
datasets = _module
augment = _module
cifar10 = _module
imagenet = _module
loader = _module
transforms = _module
models = _module
anynet = _module
blocks = _module
effnet = _module
model_zoo = _module
regnet = _module
resnet = _module
scaler = _module
vit = _module
sweep = _module
analysis = _module
htmlbook = _module
random = _module
samplers = _module
setup = _module
run_net = _module
sweep_analyze = _module
sweep_collect = _module
sweep_launch = _module
sweep_launch_job = _module
sweep_setup = _module

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


import torch.cuda.amp as amp


import random


from collections import deque


import numpy as np


import itertools


import matplotlib.pyplot as plt


from copy import deepcopy


import torch.utils.data


import re


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.nn import Module


import torch.nn as nn


from torch.nn import Dropout


import math


from torch.nn import init


from torch.nn import Parameter


class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]


class SiLU(Module):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def activation(activation_fun=None):
    """Helper for building an activation layer."""
    activation_fun = (activation_fun or cfg.MODEL.ACTIVATION_FUN).lower()
    if activation_fun == 'relu':
        return nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
    elif activation_fun == 'silu' or activation_fun == 'swish':
        try:
            return torch.nn.SiLU()
        except AttributeError:
            return SiLU()
    elif activation_fun == 'gelu':
        return torch.nn.GELU()
    else:
        raise AssertionError('Unknown MODEL.ACTIVATION_FUN: ' + activation_fun)


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, 'Only odd size kernels supported to avoid padding issues.'
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, 'Only odd size kernels supported to avoid padding issues.'
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_out * h * w
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops, params, acts = cx['flops'], cx['params'], cx['acts']
    return {'h': 1, 'w': 1, 'flops': flops, 'params': params, 'acts': acts}


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def linear_cx(cx, w_in, w_out, *, bias=False, num_locations=1):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    flops += w_in * w_out * num_locations + (w_out * num_locations if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out * num_locations
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)


def norm2d_cx(cx, w_in):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    params += 2 * w_in
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class AnyHead(Module):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, head_width, num_classes):
        super(AnyHead, self).__init__()
        self.head_width = head_width
        if head_width > 0:
            self.conv = conv2d(w_in, head_width, 1)
            self.bn = norm2d(head_width)
            self.af = activation()
            w_in = head_width
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.af(self.bn(self.conv(x))) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, head_width, num_classes):
        if head_width > 0:
            cx = conv2d_cx(cx, w_in, head_width, 1)
            cx = norm2d_cx(cx, head_width)
            w_in = head_width
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


class VanillaBlock(Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class BasicTransform(Module):
    """Basic transformation: 3x3, BN, AF, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, groups=1):
        err_str = 'Basic transform does not support w_b and groups options'
        assert w_b is None and groups == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b=None, groups=1):
        err_str = 'Basic transform does not support w_b and groups options'
        assert w_b is None and groups == 1, err_str
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBasicBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if w_in != w_out or stride != 1:
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if w_in != w_out or stride != 1:
            h, w = cx['h'], cx['w']
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, BN, AF, 3x3, BN, AF, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, groups):
        super(BottleneckTransform, self).__init__()
        s1, s3 = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        self.a = conv2d(w_in, w_b, 1, stride=s1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=s3, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b, groups):
        s1, s3 = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        cx = conv2d_cx(cx, w_in, w_b, 1, stride=s1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=s3, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if w_in != w_out or stride != 1:
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if w_in != w_out or stride != 1:
            h, w = cx['h'], cx['w']
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class ResBottleneckLinearBlock(Module):
    """Residual linear bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckLinearBlock, self).__init__()
        self.has_skip = w_in == w_out and stride == 1
        self.f = BottleneckTransform(w_in, w_out, stride, params)

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        return BottleneckTransform.complexity(cx, w_in, w_out, stride, params)


class ResStemCifar(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, 'Only odd size kernels supported to avoid padding issues.'
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def pool2d_cx(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, 'Only odd size kernels supported to avoid padding issues.'
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * h * w
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class ResStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class SimpleStem(Module):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class AnyStage(Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        for i in range(d):
            block = block_fun(w_in, w_out, stride, params)
            self.add_module('b{}'.format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            cx = block_fun.complexity(cx, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return cx


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {'vanilla_block': VanillaBlock, 'res_basic_block': ResBasicBlock, 'res_bottleneck_block': ResBottleneckBlock, 'res_bottleneck_linear_block': ResBottleneckLinearBlock}
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {'res_stem_cifar': ResStemCifar, 'res_stem_in': ResStem, 'simple_stem_in': SimpleStem}
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class AnyNet(Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {'stem_type': cfg.ANYNET.STEM_TYPE, 'stem_w': cfg.ANYNET.STEM_W, 'block_type': cfg.ANYNET.BLOCK_TYPE, 'depths': cfg.ANYNET.DEPTHS, 'widths': cfg.ANYNET.WIDTHS, 'strides': cfg.ANYNET.STRIDES, 'bot_muls': cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones, 'group_ws': cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones, 'head_w': cfg.ANYNET.HEAD_W, 'se_r': cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0, 'num_classes': cfg.MODEL.NUM_CLASSES}

    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p['stem_type'])
        block_fun = get_block_fun(p['block_type'])
        self.stem = stem_fun(3, p['stem_w'])
        prev_w = p['stem_w']
        keys = ['depths', 'widths', 'strides', 'bot_muls', 'group_ws']
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {'bot_mul': b, 'group_w': g, 'se_r': p['se_r']}
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module('s{}'.format(i + 1), stage)
            prev_w = w
        self.head = AnyHead(prev_w, p['head_w'], p['num_classes'])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p['stem_type'])
        block_fun = get_block_fun(p['block_type'])
        cx = stem_fun.complexity(cx, 3, p['stem_w'])
        prev_w = p['stem_w']
        keys = ['depths', 'widths', 'strides', 'bot_muls', 'group_ws']
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {'bot_mul': b, 'group_w': g, 'se_r': p['se_r']}
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = w
        cx = AnyHead.complexity(cx, prev_w, p['head_w'], p['num_classes'])
        return cx


class SE(Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(conv2d(w_in, w_se, 1, bias=True), activation(), conv2d(w_se, w_in, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx['h'], cx['w']
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx['h'], cx['w'] = h, w
        return cx


class MultiheadAttention(Module):
    """Multi-head Attention block from Transformer models."""

    def __init__(self, hidden_d, n_heads):
        super(MultiheadAttention, self).__init__()
        self.block = nn.MultiheadAttention(hidden_d, n_heads, batch_first=False)

    def forward(self, query, key, value, need_weights=False):
        return self.block(query=query, key=key, value=value, need_weights=need_weights)

    @staticmethod
    def complexity(cx, hidden_d, n_heads, seq_len):
        h, w = cx['h'], cx['w']
        flops, params, acts = cx['flops'], cx['params'], cx['acts']
        flops += seq_len * (hidden_d * hidden_d * 3 + hidden_d * 3)
        params += hidden_d * hidden_d * 3 + hidden_d * 3
        acts += hidden_d * 3 * seq_len
        head_d = hidden_d // n_heads
        flops += n_heads * (seq_len * head_d * seq_len)
        acts += n_heads * seq_len * seq_len
        flops += n_heads * (seq_len * seq_len * head_d)
        acts += n_heads * seq_len * head_d
        flops += seq_len * (hidden_d * hidden_d + hidden_d)
        params += hidden_d * hidden_d + hidden_d
        acts += hidden_d * seq_len
        return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class EffHead(Module):
    """EfficientNet head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, num_classes):
        super(EffHead, self).__init__()
        dropout_ratio = cfg.EN.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = activation()
        self.avg_pool = gap2d(w_out)
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if self.dropout else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class MBConv(Module):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out):
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = activation()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, 1)
            cx = norm2d_cx(cx, w_exp)
        cx = conv2d_cx(cx, w_exp, w_exp, k, stride=stride, groups=w_exp)
        cx = norm2d_cx(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffStage(Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        for i in range(d):
            block = MBConv(w_in, exp_r, k, stride, se_r, w_out)
            self.add_module('b{}'.format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out, d):
        for _ in range(d):
            cx = MBConv.complexity(cx, w_in, exp_r, k, stride, se_r, w_out)
            stride, w_in = 1, w_out
        return cx


class StemIN(Module):
    """EfficientNet stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffNet(Module):
    """EfficientNet model."""

    @staticmethod
    def get_params():
        return {'sw': cfg.EN.STEM_W, 'ds': cfg.EN.DEPTHS, 'ws': cfg.EN.WIDTHS, 'exp_rs': cfg.EN.EXP_RATIOS, 'se_r': cfg.EN.SE_R, 'ss': cfg.EN.STRIDES, 'ks': cfg.EN.KERNELS, 'hw': cfg.EN.HEAD_W, 'nc': cfg.MODEL.NUM_CLASSES}

    def __init__(self, params=None):
        super(EffNet, self).__init__()
        p = EffNet.get_params() if not params else params
        vs = ['sw', 'ds', 'ws', 'exp_rs', 'se_r', 'ss', 'ks', 'hw', 'nc']
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, sw)
        prev_w = sw
        for i, (d, w, exp_r, stride, k) in enumerate(stage_params):
            stage = EffStage(prev_w, exp_r, k, stride, se_r, w, d)
            self.add_module('s{}'.format(i + 1), stage)
            prev_w = w
        self.head = EffHead(prev_w, hw, nc)
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = EffNet.get_params() if not params else params
        vs = ['sw', 'ds', 'ws', 'exp_rs', 'se_r', 'ss', 'ks', 'hw', 'nc']
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        cx = StemIN.complexity(cx, 3, sw)
        prev_w = sw
        for d, w, exp_r, stride, k in stage_params:
            cx = EffStage.complexity(cx, prev_w, exp_r, k, stride, se_r, w, d)
            prev_w = w
        cx = EffHead.complexity(cx, prev_w, hw, nc)
        return cx


class ResHead(Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, num_classes):
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


class ResBlock(Module):
    """Residual block: x + f(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, groups=1):
        super(ResBlock, self).__init__()
        self.proj, self.bn = None, None
        if w_in != w_out or stride != 1:
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = trans_fun(w_in, w_out, stride, w_b, groups)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, trans_fun, w_b, groups):
        if w_in != w_out or stride != 1:
            h, w = cx['h'], cx['w']
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = trans_fun.complexity(cx, w_in, w_out, stride, w_b, groups)
        return cx


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {'basic_transform': BasicTransform, 'bottleneck_transform': BottleneckTransform}
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class ResStage(Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, groups=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, groups)
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, w_b=None, groups=1):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_f = get_trans_fun(cfg.RESNET.TRANS_FUN)
            cx = ResBlock.complexity(cx, b_w_in, w_out, b_stride, trans_f, w_b, groups)
        return cx


class ResStemIN(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


_IN_STAGE_DS = {(50): (3, 4, 6, 3), (101): (3, 4, 23, 3), (152): (3, 8, 36, 3)}


class ResNet(Module):
    """ResNet model."""

    def __init__(self):
        datasets = ['cifar10', 'imagenet']
        err_str = 'Dataset {} is not supported'
        assert cfg.TRAIN.DATASET in datasets, err_str.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in datasets, err_str.format(cfg.TEST.DATASET)
        super(ResNet, self).__init__()
        if 'cifar' in cfg.TRAIN.DATASET:
            self._construct_cifar()
        else:
            self._construct_imagenet()
        self.apply(init_weights)

    def _construct_cifar(self):
        err_str = 'Model depth should be of the format 6n + 2 for cifar'
        assert (cfg.MODEL.DEPTH - 2) % 6 == 0, err_str
        d = int((cfg.MODEL.DEPTH - 2) / 6)
        self.stem = ResStemCifar(3, 16)
        self.s1 = ResStage(16, 16, stride=1, d=d)
        self.s2 = ResStage(16, 32, stride=2, d=d)
        self.s3 = ResStage(32, 64, stride=2, d=d)
        self.head = ResHead(64, cfg.MODEL.NUM_CLASSES)

    def _construct_imagenet(self):
        g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
        d1, d2, d3, d4 = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, groups=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, groups=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, groups=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, groups=g)
        self.head = ResHead(2048, cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        if 'cifar' in cfg.TRAIN.DATASET:
            d = int((cfg.MODEL.DEPTH - 2) / 6)
            cx = ResStemCifar.complexity(cx, 3, 16)
            cx = ResStage.complexity(cx, 16, 16, stride=1, d=d)
            cx = ResStage.complexity(cx, 16, 32, stride=2, d=d)
            cx = ResStage.complexity(cx, 32, 64, stride=2, d=d)
            cx = ResHead.complexity(cx, 64, cfg.MODEL.NUM_CLASSES)
        else:
            g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
            d1, d2, d3, d4 = _IN_STAGE_DS[cfg.MODEL.DEPTH]
            w_b = gw * g
            cx = ResStemIN.complexity(cx, 3, 64)
            cx = ResStage.complexity(cx, 64, 256, 1, d=d1, w_b=w_b, groups=g)
            cx = ResStage.complexity(cx, 256, 512, 2, d=d2, w_b=w_b * 2, groups=g)
            cx = ResStage.complexity(cx, 512, 1024, 2, d=d3, w_b=w_b * 4, groups=g)
            cx = ResStage.complexity(cx, 1024, 2048, 2, d=d4, w_b=w_b * 8, groups=g)
            cx = ResHead.complexity(cx, 2048, cfg.MODEL.NUM_CLASSES)
        return cx


class ViTHead(Module):
    """Transformer classifier, an fc layer."""

    def __init__(self, w_in, num_classes):
        super().__init__()
        self.head_fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        return self.head_fc(x)

    @staticmethod
    def complexity(cx, w_in, num_classes):
        return linear_cx(cx, w_in, num_classes, bias=True)


class MLPBlock(Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d):
        super().__init__()
        self.linear_1 = linear(w_in, mlp_d, bias=True)
        self.af = activation('gelu')
        self.linear_2 = linear(mlp_d, w_in, bias=True)

    def forward(self, x):
        return self.linear_2(self.af(self.linear_1(x)))

    @staticmethod
    def complexity(cx, w_in, mlp_d, seq_len):
        cx = linear_cx(cx, w_in, mlp_d, bias=True, num_locations=seq_len)
        cx = linear_cx(cx, mlp_d, w_in, bias=True, num_locations=seq_len)
        return cx


def layernorm(w_in):
    """Helper for building a layernorm layer."""
    return nn.LayerNorm(w_in, eps=cfg.LN.EPS)


def layernorm_cx(cx, w_in):
    """Accumulates complexity of layernorm into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    params += 2 * w_in
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class ViTEncoderBlock(Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(self, hidden_d, n_heads, mlp_d):
        super().__init__()
        self.ln_1 = layernorm(hidden_d)
        self.self_attention = MultiheadAttention(hidden_d, n_heads)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d)

    def forward(self, x):
        x_p = self.ln_1(x)
        x_p, _ = self.self_attention(x_p, x_p, x_p)
        x = x + x_p
        x_p = self.mlp_block(self.ln_2(x))
        return x + x_p

    @staticmethod
    def complexity(cx, hidden_d, n_heads, mlp_d, seq_len):
        cx = layernorm_cx(cx, hidden_d)
        cx = MultiheadAttention.complexity(cx, hidden_d, n_heads, seq_len)
        cx = layernorm_cx(cx, hidden_d)
        cx = MLPBlock.complexity(cx, hidden_d, mlp_d, seq_len)
        return cx


class ViTEncoder(Module):
    """Transformer encoder (sequence of ViTEncoderBlocks)."""

    def __init__(self, n_layers, hidden_d, n_heads, mlp_d):
        super(ViTEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(f'block_{i}', ViTEncoderBlock(hidden_d, n_heads, mlp_d))
        self.ln = layernorm(hidden_d)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, n_layers, hidden_d, n_heads, mlp_d, seq_len):
        for _ in range(n_layers):
            cx = ViTEncoderBlock.complexity(cx, hidden_d, n_heads, mlp_d, seq_len)
        cx = layernorm_cx(cx, hidden_d)
        return cx


def patchify2d(w_in, w_out, k, *, bias=True):
    """Helper for building a patchify layer as used by ViT models."""
    return nn.Conv2d(w_in, w_out, k, stride=k, padding=0, bias=bias)


def patchify2d_cx(cx, w_in, w_out, k, *, bias=True):
    """Accumulates complexity of patchify2d into cx = (h, w, flops, params, acts)."""
    err_str = 'Only kernel sizes divisible by the input size are supported.'
    assert cx['h'] % k == 0 and cx['w'] % k == 0, err_str
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'], cx['acts']
    h, w = h // k, w // k
    flops += k * k * w_in * w_out * h * w + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out + (w_out if bias else 0)
    acts += w_out * h * w
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class ViTStemPatchify(Module):
    """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, w_in, w_out, k):
        super(ViTStemPatchify, self).__init__()
        self.patchify = patchify2d(w_in, w_out, k, bias=True)

    def forward(self, x):
        return self.patchify(x)

    @staticmethod
    def complexity(cx, w_in, w_out, k):
        return patchify2d_cx(cx, w_in, w_out, k, bias=True)


class ViTStemConv(Module):
    """The conv vision transformer stem as per https://arxiv.org/abs/2106.14881."""

    def __init__(self, w_in, ks, ws, ss):
        super(ViTStemConv, self).__init__()
        for i, (k, w_out, stride) in enumerate(zip(ks, ws, ss)):
            if i < len(ks) - 1:
                self.add_module(f'cstem{i}_conv', conv2d(w_in, w_out, 3, stride=stride))
                self.add_module(f'cstem{i}_bn', norm2d(w_out))
                self.add_module(f'cstem{i}_af', activation('relu'))
            else:
                m = conv2d(w_in, w_out, k, stride=stride, bias=True)
                self.add_module('cstem_last_conv', m)
            w_in = w_out

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, ks, ws, ss):
        for i, (k, w_out, stride) in enumerate(zip(ks, ws, ss)):
            if i < len(ks) - 1:
                cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
                cx = norm2d_cx(cx, w_out)
            else:
                cx = conv2d_cx(cx, w_in, w_out, k, stride=stride, bias=True)
            w_in = w_out
        return cx


def init_weights_vit(model):
    """Performs ViT weight init."""
    for k, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if 'patchify' in k:
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                init.trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
                init.zeros_(m.bias)
            elif 'cstem_last' in k:
                init.normal_(m.weight, mean=0.0, std=math.sqrt(1.0 / m.out_channels))
                init.zeros_(m.bias)
            elif 'cstem' in k:
                init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / m.out_channels))
            else:
                raise NotImplementedError
        if isinstance(m, torch.nn.Linear):
            if 'self_attention' in k:
                pass
            elif 'mlp_block' in k:
                init.xavier_uniform_(m.weight)
                init.normal_(m.bias, std=1e-06)
            elif 'head_fc' in k:
                init.zeros_(m.weight)
                init.zeros_(m.bias)
            else:
                raise NotImplementedError
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.LayerNorm):
            pass
    init.normal_(model.pos_embedding, mean=0.0, std=0.02)


class ViT(Module):
    """Vision transformer as per https://arxiv.org/abs/2010.11929."""

    @staticmethod
    def get_params():
        return {'image_size': cfg.TRAIN.IM_SIZE, 'patch_size': cfg.VIT.PATCH_SIZE, 'stem_type': cfg.VIT.STEM_TYPE, 'c_stem_kernels': cfg.VIT.C_STEM_KERNELS, 'c_stem_strides': cfg.VIT.C_STEM_STRIDES, 'c_stem_dims': cfg.VIT.C_STEM_DIMS, 'n_layers': cfg.VIT.NUM_LAYERS, 'n_heads': cfg.VIT.NUM_HEADS, 'hidden_d': cfg.VIT.HIDDEN_DIM, 'mlp_d': cfg.VIT.MLP_DIM, 'cls_type': cfg.VIT.CLASSIFIER_TYPE, 'num_classes': cfg.MODEL.NUM_CLASSES}

    @staticmethod
    def check_params(params):
        p = params
        err_str = 'Input shape indivisible by patch size'
        assert p['image_size'] % p['patch_size'] == 0, err_str
        assert p['stem_type'] in ['patchify', 'conv'], 'Unexpected stem type'
        assert p['cls_type'] in ['token', 'pooled'], 'Unexpected classifier mode'
        if p['stem_type'] == 'conv':
            err_str = 'Conv stem layers mismatch'
            assert len(p['c_stem_dims']) == len(p['c_stem_strides']), err_str
            assert len(p['c_stem_strides']) == len(p['c_stem_kernels']), err_str
            err_str = 'Stem strides unequal to patch size'
            assert p['patch_size'] == np.prod(p['c_stem_strides']), err_str
            err_str = 'Stem output dim unequal to hidden dim'
            assert p['c_stem_dims'][-1] == p['hidden_d'], err_str

    def __init__(self, params=None):
        super(ViT, self).__init__()
        p = ViT.get_params() if not params else params
        ViT.check_params(p)
        if p['stem_type'] == 'patchify':
            self.stem = ViTStemPatchify(3, p['hidden_d'], p['patch_size'])
        elif p['stem_type'] == 'conv':
            ks, ws, ss = p['c_stem_kernels'], p['c_stem_dims'], p['c_stem_strides']
            self.stem = ViTStemConv(3, ks, ws, ss)
        seq_len = (p['image_size'] // cfg.VIT.PATCH_SIZE) ** 2
        if p['cls_type'] == 'token':
            self.class_token = Parameter(torch.zeros(1, 1, p['hidden_d']))
            seq_len += 1
        else:
            self.class_token = None
        self.pos_embedding = Parameter(torch.zeros(seq_len, 1, p['hidden_d']))
        self.encoder = ViTEncoder(p['n_layers'], p['hidden_d'], p['n_heads'], p['mlp_d'])
        self.head = ViTHead(p['hidden_d'], p['num_classes'])
        init_weights_vit(self)

    def forward(self, x):
        x = self.stem(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(2, 0, 1)
        if self.class_token is not None:
            class_token = self.class_token.expand(-1, x.size(1), -1)
            x = torch.cat([class_token, x], dim=0)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x[0, :, :] if self.class_token is not None else x.mean(dim=0)
        return self.head(x)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity. If you alter the model, make sure to update."""
        p = ViT.get_params() if not params else params
        ViT.check_params(p)
        if p['stem_type'] == 'patchify':
            cx = ViTStemPatchify.complexity(cx, 3, p['hidden_d'], p['patch_size'])
        elif p['stem_type'] == 'conv':
            ks, ws, ss = p['c_stem_kernels'], p['c_stem_dims'], p['c_stem_strides']
            cx = ViTStemConv.complexity(cx, 3, ks, ws, ss)
        seq_len = (p['image_size'] // cfg.VIT.PATCH_SIZE) ** 2
        if p['cls_type'] == 'token':
            seq_len += 1
            cx['params'] += p['hidden_d']
        cx['params'] += seq_len * p['hidden_d']
        cx = ViTEncoder.complexity(cx, p['n_layers'], p['hidden_d'], p['n_heads'], p['mlp_d'], seq_len)
        cx = ViTHead.complexity(cx, p['hidden_d'], p['num_classes'])
        return cx


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'd': 4, 'block_fun': _mock_layer, 'params': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPBlock,
     lambda: ([], {'w_in': 4, 'mlp_d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'hidden_d': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (ResHead,
     lambda: ([], {'w_in': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ViTHead,
     lambda: ([], {'w_in': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ViTStemPatchify,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_pycls(_paritybench_base):
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

