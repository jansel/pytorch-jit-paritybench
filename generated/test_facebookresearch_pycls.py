import sys
_module = sys.modules[__name__]
del sys
test_complexity = _module
pycls = _module
core = _module
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
cifar10 = _module
imagenet = _module
loader = _module
transforms = _module
models = _module
anynet = _module
effnet = _module
regnet = _module
resnet = _module
setup = _module
test_net = _module
time_net = _module
train_net = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import itertools


import math


import torch.nn as nn


import numpy as np


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx['h'], cx['w'] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx


_global_config['BN'] = 4


_global_config['MEM'] = 4


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = 'Vanilla block does not support bm, gw, and se_r options'
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False
            )
        self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = 'Vanilla block does not support bm, gw, and se_r options'
        assert bm is None and gw is None and se_r is None, err_str
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False
            )
        self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = 'Basic transform does not support bm, gw, and se_r options'
        assert bm is None and gw is None and se_r is None, err_str
        super(ResBasicBlock, self).__init__()
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0,
                bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
                )
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = 'Basic transform does not support bm, gw, and se_r options'
        assert bm is None and gw is None and se_r is None, err_str
        proj_block = w_in != w_out or stride != 1
        if proj_block:
            h, w = cx['h'], cx['w']
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride)
        return cx


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, 1, bias=True), nn.
            ReLU(inplace=cfg.MEM.RELU_INPLACE), nn.Conv2d(w_se, w_in, 1,
            bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx['h'], cx['w']
        cx['h'], cx['w'] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
        cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
        cx['h'], cx['w'] = h, w
        return cx


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g,
            bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        g = w_b // gw
        cx = net.complexity_conv2d(cx, w_in, w_b, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_b)
        cx = net.complexity_conv2d(cx, w_b, w_b, 3, stride, 1, g)
        cx = net.complexity_batchnorm2d(cx, w_b)
        if se_r:
            w_se = int(round(w_in * se_r))
            cx = SE.complexity(cx, w_b, w_se)
        cx = net.complexity_conv2d(cx, w_b, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0,
                bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
                )
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        proj_block = w_in != w_out or stride != 1
        if proj_block:
            h, w = cx['h'], cx['w']
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, bm, gw,
            se_r)
        return cx


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 7, 2, 3)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_maxpool2d(cx, 3, 2, 1)
        return cx


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw,
                se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            cx = block_fun.complexity(cx, b_w_in, w_out, b_stride, bm, gw, se_r
                )
        return cx


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {'res_stem_cifar': ResStemCifar, 'res_stem_in': ResStemIN,
        'simple_stem_in': SimpleStemIN}
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {'vanilla_block': VanillaBlock, 'res_basic_block':
        ResBasicBlock, 'res_bottleneck_block': ResBottleneckBlock}
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


_global_config['MODEL'] = 4


_global_config['ANYNET'] = 4


class AnyNet(nn.Module):
    """AnyNet model."""

    @staticmethod
    def get_args():
        return {'stem_type': cfg.ANYNET.STEM_TYPE, 'stem_w': cfg.ANYNET.
            STEM_W, 'block_type': cfg.ANYNET.BLOCK_TYPE, 'ds': cfg.ANYNET.
            DEPTHS, 'ws': cfg.ANYNET.WIDTHS, 'ss': cfg.ANYNET.STRIDES,
            'bms': cfg.ANYNET.BOT_MULS, 'gws': cfg.ANYNET.GROUP_WS, 'se_r':
            cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else None, 'nc': cfg.MODEL.
            NUM_CLASSES}

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        self._construct(**kwargs)
        self.apply(net.init_weights)

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms,
        gws, se_r, nc):
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm,
                gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, **kwargs):
        """Computes model complexity. If you alter the model, make sure to update."""
        kwargs = AnyNet.get_args() if not kwargs else kwargs
        return AnyNet._complexity(cx, **kwargs)

    @staticmethod
    def _complexity(cx, stem_type, stem_w, block_type, ds, ws, ss, bms, gws,
        se_r, nc):
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        cx = stem_fun.complexity(cx, 3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for d, w, s, bm, gw in stage_params:
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, bm, gw,
                se_r)
            prev_w = w
        cx = AnyHead.complexity(cx, prev_w, nc)
        return cx


_global_config['EN'] = 4


class EffHead(nn.Module):
    """EfficientNet head: 1x1, BN, Swish, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, nc):
        super(EffHead, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.
            BN.MOM)
        self.conv_swish = Swish()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if cfg.EN.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EN.DROPOUT_RATIO)
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, 'dropout') else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, nc):
        cx = net.complexity_conv2d(cx, w_in, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx['h'], cx['w'] = 1, 1
        cx = net.complexity_conv2d(cx, w_out, nc, 1, 1, 0, bias=True)
        return cx


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, 1, bias=True),
            Swish(), nn.Conv2d(w_se, w_in, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx['h'], cx['w']
        cx['h'], cx['w'] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
        cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
        cx['h'], cx['w'] = h, w
        return cx


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1, padding=0, bias=
                False)
            self.exp_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=
                cfg.BN.MOM)
            self.exp_swish = Swish()
        dwise_args = {'groups': w_exp, 'padding': (kernel - 1) // 2, 'bias':
            False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel, stride=stride, **
            dwise_args)
        self.dwise_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=cfg.
            BN.MOM)
        self.dwise_swish = Swish()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(w_exp, w_out, 1, stride=1, padding=0,
            bias=False)
        self.lin_proj_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=
            cfg.BN.MOM)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = net.drop_connect(f_x, cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = net.complexity_conv2d(cx, w_in, w_exp, 1, 1, 0)
            cx = net.complexity_batchnorm2d(cx, w_exp)
        padding = (kernel - 1) // 2
        cx = net.complexity_conv2d(cx, w_exp, w_exp, kernel, stride,
            padding, w_exp)
        cx = net.complexity_batchnorm2d(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = net.complexity_conv2d(cx, w_exp, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = 'b{}'.format(i + 1)
            self.add_module(name, MBConv(b_w_in, exp_r, kernel, b_stride,
                se_r, w_out))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out, d):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            cx = MBConv.complexity(cx, b_w_in, exp_r, kernel, b_stride,
                se_r, w_out)
        return cx


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


_global_config['TRAIN'] = 4


_global_config['TEST'] = 4


class EffNet(nn.Module):
    """EfficientNet model."""

    @staticmethod
    def get_args():
        return {'stem_w': cfg.EN.STEM_W, 'ds': cfg.EN.DEPTHS, 'ws': cfg.EN.
            WIDTHS, 'exp_rs': cfg.EN.EXP_RATIOS, 'se_r': cfg.EN.SE_R, 'ss':
            cfg.EN.STRIDES, 'ks': cfg.EN.KERNELS, 'head_w': cfg.EN.HEAD_W,
            'nc': cfg.MODEL.NUM_CLASSES}

    def __init__(self):
        err_str = 'Dataset {} is not supported'
        assert cfg.TRAIN.DATASET in ['imagenet'], err_str.format(cfg.TRAIN.
            DATASET)
        assert cfg.TEST.DATASET in ['imagenet'], err_str.format(cfg.TEST.
            DATASET)
        super(EffNet, self).__init__()
        self._construct(**EffNet.get_args())
        self.apply(net.init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            self.add_module(name, EffStage(prev_w, exp_r, kernel, stride,
                se_r, w, d))
            prev_w = w
        self.head = EffHead(prev_w, head_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        return EffNet._complexity(cx, **EffNet.get_args())

    @staticmethod
    def _complexity(cx, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        cx = StemIN.complexity(cx, 3, stem_w)
        prev_w = stem_w
        for d, w, exp_r, stride, kernel in stage_params:
            cx = EffStage.complexity(cx, prev_w, exp_r, kernel, stride,
                se_r, w, d)
            prev_w = w
        cx = EffHead.complexity(cx, prev_w, head_w, nc)
        return cx


class ResHead(nn.Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx['h'], cx['w'] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = 'Basic transform does not support w_b and num_gs options'
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False
            )
        self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = 'Basic transform does not support w_b and num_gs options'
        assert w_b is None and num_gs == 1, err_str
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


_global_config['RESNET'] = 4


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        s1, s3 = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        self.a = nn.Conv2d(w_in, w_b, 1, stride=s1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs,
            bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b, num_gs):
        s1, s3 = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        cx = net.complexity_conv2d(cx, w_in, w_b, 1, s1, 0)
        cx = net.complexity_batchnorm2d(cx, w_b)
        cx = net.complexity_conv2d(cx, w_b, w_b, 3, s3, 1, num_gs)
        cx = net.complexity_batchnorm2d(cx, w_b)
        cx = net.complexity_conv2d(cx, w_b, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0,
                bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
                )
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, trans_fun, w_b, num_gs):
        proj_block = w_in != w_out or stride != 1
        if proj_block:
            h, w = cx['h'], cx['w']
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx['h'], cx['w'] = h, w
        cx = trans_fun.complexity(cx, w_in, w_out, stride, w_b, num_gs)
        return cx


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {'basic_transform': BasicTransform, 'bottleneck_transform':
        BottleneckTransform}
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b,
                num_gs)
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, w_b=None, num_gs=1):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_f = get_trans_fun(cfg.RESNET.TRANS_FUN)
            cx = ResBlock.complexity(cx, b_w_in, w_out, b_stride, trans_f,
                w_b, num_gs)
        return cx


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 7, 2, 3)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_maxpool2d(cx, 3, 2, 1)
        return cx


_IN_STAGE_DS = {(50): (3, 4, 6, 3), (101): (3, 4, 23, 3), (152): (3, 8, 36, 3)}


class ResNet(nn.Module):
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
        self.apply(net.init_weights)

    def _construct_cifar(self):
        err_str = 'Model depth should be of the format 6n + 2 for cifar'
        assert (cfg.MODEL.DEPTH - 2) % 6 == 0, err_str
        d = int((cfg.MODEL.DEPTH - 2) / 6)
        self.stem = ResStemCifar(3, 16)
        self.s1 = ResStage(16, 16, stride=1, d=d)
        self.s2 = ResStage(16, 32, stride=2, d=d)
        self.s3 = ResStage(32, 64, stride=2, d=d)
        self.head = ResHead(64, nc=cfg.MODEL.NUM_CLASSES)

    def _construct_imagenet(self):
        g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
        d1, d2, d3, d4 = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)
        self.head = ResHead(2048, nc=cfg.MODEL.NUM_CLASSES)

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
            cx = ResHead.complexity(cx, 64, nc=cfg.MODEL.NUM_CLASSES)
        else:
            g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
            d1, d2, d3, d4 = _IN_STAGE_DS[cfg.MODEL.DEPTH]
            w_b = gw * g
            cx = ResStemIN.complexity(cx, 3, 64)
            cx = ResStage.complexity(cx, 64, 256, 1, d=d1, w_b=w_b, num_gs=g)
            cx = ResStage.complexity(cx, 256, 512, 2, d=d2, w_b=w_b * 2,
                num_gs=g)
            cx = ResStage.complexity(cx, 512, 1024, 2, d=d3, w_b=w_b * 4,
                num_gs=g)
            cx = ResStage.complexity(cx, 1024, 2048, 2, d=d4, w_b=w_b * 8,
                num_gs=g)
            cx = ResHead.complexity(cx, 2048, nc=cfg.MODEL.NUM_CLASSES)
        return cx


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_pycls(_paritybench_base):
    pass
    def test_000(self):
        self._check(AnyHead(*[], **{'w_in': 4, 'nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SE(*[], **{'w_in': 4, 'w_se': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ResHead(*[], **{'w_in': 4, 'nc': 4}), [torch.rand([4, 4, 4, 4])], {})

