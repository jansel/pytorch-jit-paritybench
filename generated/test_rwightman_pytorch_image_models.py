import sys
_module = sys.modules[__name__]
del sys
avg_checkpoints = _module
clean_checkpoint = _module
convert_from_mxnet = _module
hubconf = _module
inference = _module
setup = _module
sotabench = _module
tests = _module
test_layers = _module
test_models = _module
timm = _module
data = _module
auto_augment = _module
config = _module
constants = _module
dataset = _module
distributed_sampler = _module
loader = _module
mixup = _module
random_erasing = _module
tf_preprocessing = _module
transforms = _module
transforms_factory = _module
loss = _module
cross_entropy = _module
jsd = _module
models = _module
densenet = _module
dla = _module
dpn = _module
efficientnet = _module
efficientnet_blocks = _module
efficientnet_builder = _module
factory = _module
feature_hooks = _module
gluon_resnet = _module
gluon_xception = _module
helpers = _module
hrnet = _module
inception_resnet_v2 = _module
inception_v3 = _module
inception_v4 = _module
layers = _module
activations = _module
activations_jit = _module
activations_me = _module
adaptive_avgmax_pool = _module
anti_aliasing = _module
blur_pool = _module
cbam = _module
cond_conv2d = _module
conv2d_same = _module
conv_bn_act = _module
create_act = _module
create_attn = _module
create_conv2d = _module
create_norm_act = _module
drop = _module
eca = _module
evo_norm = _module
inplace_abn = _module
median_pool = _module
mixed_conv2d = _module
norm_act = _module
padding = _module
pool2d_same = _module
se = _module
selective_kernel = _module
separable_conv = _module
space_to_depth = _module
split_attn = _module
split_batchnorm = _module
test_time_pool = _module
weight_init = _module
mobilenetv3 = _module
nasnet = _module
pnasnet = _module
registry = _module
regnet = _module
res2net = _module
resnest = _module
resnet = _module
selecsls = _module
senet = _module
sknet = _module
tresnet = _module
vovnet = _module
xception = _module
optim = _module
adamw = _module
lookahead = _module
nadam = _module
novograd = _module
nvnovograd = _module
optim_factory = _module
radam = _module
rmsprop_tf = _module
scheduler = _module
cosine_lr = _module
plateau_lr = _module
scheduler_factory = _module
step_lr = _module
tanh_lr = _module
utils = _module
version = _module
train = _module
validate = _module

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


import logging


import numpy as np


import torch


import torch.nn as nn


import math


from torch.utils.data import Sampler


import torch.distributed as dist


import torch.nn.functional as F


import re


from collections import OrderedDict


from functools import partial


import torch.utils.checkpoint as cp


from torch.jit.annotations import List


from typing import Tuple


from typing import List


from torch.nn import functional as F


from copy import deepcopy


import torch.utils.model_zoo as model_zoo


from torch import nn as nn


import torch.nn.parallel


from typing import Dict


from typing import Optional


import types


import functools


from torch import nn


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _quadruple


_SCRIPTABLE = False


def is_scriptable():
    return _SCRIPTABLE


_EXPORTABLE = False


def is_exportable():
    return _EXPORTABLE


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by Ross Wightman
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-07, 1
            ).log()
        loss += self.alpha * sum([F.kl_div(logp_mixture, p_split, reduction
            ='batchmean') for p_split in probs]) / len(probs)
        return loss


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        norm_layer=nn.ReLU, drop_rate=0.0, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                growth_rate=growth_rate, bn_size=bn_size, norm_layer=
                norm_layer, drop_rate=drop_rate, memory_efficient=
                memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_layer=
        nn.BatchNorm2d, aa_layer=None):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DlaBasic(nn.Module):
    """DLA Basic"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1,
        cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) *
            cardinality)
        mid_planes = mid_planes // self.expansion
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
            stride=stride, padding=dilation, bias=False, dilation=dilation,
            groups=cardinality)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False
            )
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4,
        cardinality=8, base_width=4):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) *
            cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes
        self.conv1 = nn.Conv2d(inplanes, mid_planes * scale, kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes * scale)
        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False))
            bns.append(nn.BatchNorm2d(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(mid_planes * scale, outplanes, kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = spx[i] if i == 0 or self.is_first else sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            spo.append(self.pool(spx[-1]) if self.is_first else spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class DlaRoot(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=
            False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class DlaTree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1,
        dilation=1, cardinality=1, base_width=64, level_root=False,
        root_dim=0, root_kernel_size=1, root_residual=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width
            =base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size,
                root_residual=root_residual))
            self.tree1 = DlaTree(levels - 1, block, in_channels,
                out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(levels - 1, block, out_channels,
                out_channels, root_dim=root_dim + out_channels, **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size,
                root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = nn.MaxPool2d(stride, stride=stride
            ) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=1, bias=False), nn.
                BatchNorm2d(out_channels))
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):

    def __init__(self, levels, channels, num_classes=1000, in_chans=3,
        cardinality=1, base_width=64, block=DlaBottle2neck, residual_root=
        False, linear_root=False, drop_rate=0.0, global_pool='avg'):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.base_layer = nn.Sequential(nn.Conv2d(in_chans, channels[0],
            kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d
            (channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0]
            )
        self.level1 = self._make_conv_level(channels[0], channels[1],
            levels[1], stride=2)
        cargs = dict(cardinality=cardinality, base_width=base_width,
            root_residual=residual_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2,
            level_root=False, **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2,
            level_root=True, **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2,
            level_root=True, **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2,
            level_root=True, **cargs)
        self.num_features = channels[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Conv2d(self.num_features * self.global_pool.feat_mult(
            ), num_classes, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3,
                stride=stride if i == 0 else 1, padding=dilation, bias=
                False, dilation=dilation), nn.BatchNorm2d(planes), nn.ReLU(
                inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.fc = nn.Conv2d(num_features, num_classes, kernel_size=1,
                bias=True)
        else:
            self.fc = nn.Identity()

    def forward_features(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x.flatten(1)


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0,
        groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding,
            groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):

    def __init__(self, num_init_features, kernel_size=7, in_chans=3,
        padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, num_init_features, kernel_size=
            kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups,
        block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False
        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a,
            kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b,
            kernel_size=3, stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1,
                bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c +
                inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            if self.c1x1_w_s1 is not None:
                x_s = self.c1x1_w_s1(x_in)
            else:
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
        b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg',
        fc_act=nn.ELU()):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.b = b
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, in_chans=
                in_chans, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, in_chans=
                in_chans, kernel_size=7, padding=3)
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc,
            groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs, activation_fn=fc_act)
        self.num_features = in_chs
        self.features = nn.Sequential(blocks)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_features = self.num_features * self.global_pool.feat_mult()
        self.classifier = nn.Conv2d(num_features, num_classes, kernel_size=
            1, bias=True)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classifier = nn.Conv2d(num_features, num_classes,
                kernel_size=1, bias=True)
        else:
            self.classifier = nn.Identity()

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        out = self.classifier(x)
        return out.flatten(1)


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
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.
        dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def is_static_pad(kernel_size: int, stride: int=1, dilation: int=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def get_padding(kernel_size, stride, dilation=1):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def get_padding_value(padding, kernel_size, **kwargs) ->Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **
            kwargs)


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [
            kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits,
            out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(str(idx), create_conv2d_pad(in_ch, out_ch, k,
                stride=stride, padding=padding, dilation=dilation, groups=
                conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs
        assert 'groups' not in kwargs
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=
                groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size,
                groups=groups, **kwargs)
    return m


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


class FeatureHooks:

    def __init__(self, hooks, named_modules):
        modules = {k: v for k, v in named_modules}
        for h in hooks:
            hook_name = h['name']
            m = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            if h['type'] == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif h['type'] == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, 'Unsupported hook type'
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, name, *args):
        x = args[-1]
        if isinstance(x, tuple):
            x = x[0]
        self._feature_outputs[x.device][name] = x

    def get_output(self, device) ->List[torch.tensor]:
        output = list(self._feature_outputs[device].values())
        self._feature_outputs[device] = OrderedDict()
        return output


_DEBUG = False


def get_condconv_initializer(initializer, num_experts, expert_shape):

    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if len(weight.shape) != 2 or weight.shape[0
            ] != num_experts or weight.shape[1] != num_params:
            raise ValueError(
                'CondConv variables must have shape [num_experts, num_params]')
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(lambda w: w.data.normal_(
            0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4),
        feature_location='bottleneck', in_chans=3, stem_size=32,
        channel_multiplier=1.0, channel_divisor=8, channel_min=None,
        output_stride=32, pad_type='', fix_stem=False, act_layer=nn.ReLU,
        drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None, norm_layer=nn.
        BatchNorm2d, norm_kwargs=None):
        super(EfficientNetFeatures, self).__init__()
        norm_kwargs = norm_kwargs or {}
        num_stages = max(out_indices) + 1
        self.out_indices = out_indices
        self.feature_location = feature_location
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier,
                channel_divisor, channel_min)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size
        builder = EfficientNetBuilder(channel_multiplier, channel_divisor,
            channel_min, output_stride, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, feature_location=
            feature_location, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self._feature_info = builder.features
        self._stage_to_feature_idx = {v['stage_idx']: fi for fi, v in self.
            _feature_info.items() if fi in self.out_indices}
        self._in_chs = builder.in_chs
        efficientnet_init_weights(self)
        if _DEBUG:
            for k, v in self._feature_info.items():
                None
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = [dict(name=self._feature_info[idx]['module'], type=self
                ._feature_info[idx]['hook_type']) for idx in out_indices]
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def feature_channels(self, idx=None):
        """ Feature Channel Shortcut
        Returns feature channel count for each output index if idx == None. If idx is an integer, will
        return feature channel count for that feature block index (independent of out_indices setting).
        """
        if isinstance(idx, int):
            return self._feature_info[idx]['num_chs']
        return [self._feature_info[i]['num_chs'] for i in self.out_indices]

    def feature_info(self, idx=None):
        """ Feature Channel Shortcut
        Returns feature channel count for each output index if idx == None. If idx is an integer, will
        return feature channel count for that feature block index (independent of out_indices setting).
        """
        if isinstance(idx, int):
            return self._feature_info[idx]
        return [self._feature_info[i] for i in self.out_indices]

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i in self._stage_to_feature_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            return self.feature_hooks.get_output(x.device)


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(
            g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


def sigmoid(x, inplace: bool=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
        act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) *
            se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=1,
        pad_type='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
        norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=
            stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            info = dict(module='act1', hook_type='forward', num_chs=self.
                conv.out_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels
                )
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


_SE_ARGS_DEFAULT = dict(gate_fn=sigmoid, act_layer=None, reduce_mid=False,
    divisor=1)


def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
        pw_kernel_size=1, pw_act=False, se_ratio=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.drop_path_rate = drop_path_rate
        self.conv_dw = create_conv2d(in_chs, in_chs, dw_kernel_size, stride
            =stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size,
            padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True
            ) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.
                conv_pw.in_channels)
        elif location == 'depthwise':
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=
                self.conv_pw.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pw.
                out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio
        =1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_kwargs=
        None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_kwargs=None,
        drop_path_rate=0.0):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size,
            padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self.conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size,
            stride=stride, dilation=dilation, padding=pad_type, depthwise=
            True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size,
            padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.
                conv_pw.in_channels)
        elif location == 'depthwise':
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs
                =self.conv_pwl.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.
                out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0,
        fake_in_chs=0, stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
        noskip=False, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        if fake_in_chs > 0:
            mid_chs = make_divisible(fake_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size,
            padding=pad_type)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size,
            stride=stride, dilation=dilation, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.
                conv_exp.out_channels)
        elif location == 'depthwise':
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs
                =self.conv_pwl.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.
                out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


def _fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return [pad_beg, pad_end, pad_beg, pad_end]


_USE_FIXED_PAD = False


def _pytorch_padding(kernel_size, stride=1, dilation=1, **_):
    if _USE_FIXED_PAD:
        return 0
    else:
        padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
        fp = _fixed_padding(kernel_size, dilation)
        assert all(padding == p for p in fp)
        return padding


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False, norm_layer=None, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = _fixed_padding(self.kernel_size, self.dilation)
        if _USE_FIXED_PAD and any(p > 0 for p in padding):
            self.fixed_padding = nn.ZeroPad2d(padding)
        else:
            self.fixed_padding = None
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size, stride=
            stride, padding=_pytorch_padding(kernel_size, stride, dilation),
            dilation=dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(num_features=inplanes, **norm_kwargs)
        self.conv_pw = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        if self.fixed_padding is not None:
            x = self.fixed_padding(x)
        x = self.conv_dw(x)
        x = self.bn(x)
        x = self.conv_pw(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, num_reps, stride=1, dilation=1,
        norm_layer=None, norm_kwargs=None, start_with_relu=True, grow_first
        =True, is_last=False):
        super(Block, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if planes != inplanes or stride != 1:
            self.skip = nn.Sequential()
            self.skip.add_module('conv1', nn.Conv2d(inplanes, planes, 1,
                stride=stride, bias=False)),
            self.skip.add_module('bn1', norm_layer(num_features=planes, **
                norm_kwargs))
        else:
            self.skip = None
        rep = OrderedDict()
        l = 1
        filters = inplanes
        if grow_first:
            if start_with_relu:
                rep['act%d' % l] = nn.ReLU(inplace=False)
            rep['conv%d' % l] = SeparableConv2d(inplanes, planes, 3, 1,
                dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            filters = planes
            l += 1
        for _ in range(num_reps - 1):
            if grow_first or start_with_relu:
                rep['act%d' % l] = nn.ReLU(inplace=grow_first or not
                    start_with_relu)
            rep['conv%d' % l] = SeparableConv2d(filters, filters, 3, 1,
                dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=filters, **norm_kwargs)
            l += 1
        if not grow_first:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(inplanes, planes, 3, 1,
                dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1
        if stride != 1:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(planes, planes, 3, stride,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1
        elif is_last:
            rep['act%d' % l] = nn.ReLU(inplace=True)
            rep['conv%d' % l] = SeparableConv2d(planes, planes, 3, 1,
                dilation, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            rep['bn%d' % l] = norm_layer(num_features=planes, **norm_kwargs)
            l += 1
        self.rep = nn.Sequential(rep)

    def forward(self, x):
        skip = x
        if self.skip is not None:
            skip = self.skip(skip)
        x = self.rep(x) + skip
        return x


class Xception65(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_rate=0.0,
        global_pool='avg'):
        super(Xception65, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2,
            padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn2 = norm_layer(num_features=64)
        self.block1 = Block(64, 128, num_reps=2, stride=2, norm_layer=
            norm_layer, norm_kwargs=norm_kwargs, start_with_relu=False)
        self.block2 = Block(128, 256, num_reps=2, stride=2, norm_layer=
            norm_layer, norm_kwargs=norm_kwargs, start_with_relu=False,
            grow_first=True)
        self.block3 = Block(256, 728, num_reps=2, stride=
            entry_block3_stride, norm_layer=norm_layer, norm_kwargs=
            norm_kwargs, start_with_relu=True, grow_first=True, is_last=True)
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(728, 
            728, num_reps=3, stride=1, dilation=middle_block_dilation,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =True, grow_first=True)) for i in range(4, 20)]))
        self.block20 = Block(728, 1024, num_reps=2, stride=
            exit_block20_stride, dilation=exit_block_dilations[0],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer, norm_kwargs=
            norm_kwargs)
        self.bn3 = norm_layer(num_features=1536, **norm_kwargs)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer, norm_kwargs=
            norm_kwargs)
        self.bn4 = norm_layer(num_features=1536, **norm_kwargs)
        self.num_features = 2048
        self.conv5 = SeparableConv2d(1536, self.num_features, 3, stride=1,
            dilation=exit_block_dilations[1], norm_layer=norm_layer,
            norm_kwargs=norm_kwargs)
        self.bn5 = norm_layer(num_features=self.num_features, **norm_kwargs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.mid(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class Xception71(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_rate=0.0,
        global_pool='avg'):
        super(Xception71, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2,
            padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn2 = norm_layer(num_features=64)
        self.block1 = Block(64, 128, num_reps=2, stride=2, norm_layer=
            norm_layer, norm_kwargs=norm_kwargs, start_with_relu=False)
        self.block2 = nn.Sequential(*[Block(128, 256, num_reps=2, stride=1,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =False, grow_first=True), Block(256, 256, num_reps=2, stride=2,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =False, grow_first=True), Block(256, 728, num_reps=2, stride=2,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =False, grow_first=True)])
        self.block3 = Block(728, 728, num_reps=2, stride=
            entry_block3_stride, norm_layer=norm_layer, norm_kwargs=
            norm_kwargs, start_with_relu=True, grow_first=True, is_last=True)
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(728, 
            728, num_reps=3, stride=1, dilation=middle_block_dilation,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =True, grow_first=True)) for i in range(4, 20)]))
        self.block20 = Block(728, 1024, num_reps=2, stride=
            exit_block20_stride, dilation=exit_block_dilations[0],
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, start_with_relu
            =True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer, norm_kwargs=
            norm_kwargs)
        self.bn3 = norm_layer(num_features=1536, **norm_kwargs)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer, norm_kwargs=
            norm_kwargs)
        self.bn4 = norm_layer(num_features=1536, **norm_kwargs)
        self.num_features = 2048
        self.conv5 = SeparableConv2d(1536, self.num_features, 3, stride=1,
            dilation=exit_block_dilations[1], norm_layer=norm_layer,
            norm_kwargs=norm_kwargs)
        self.bn5 = norm_layer(num_features=self.num_features, **norm_kwargs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.fc = nn.Linear(num_features, num_classes)
        else:
            self.fc = nn.Identity()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.mid(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


logger = logging.getLogger(__name__)


_BN_MOMENTUM = 0.1


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks,
            num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                num_channels[branch_index] * block.expansion, momentum=
                _BN_MOMENTUM))
        layers = [block(self.num_inchannels[branch_index], num_channels[
            branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, 1, 0, bias
                        =False), nn.BatchNorm2d(num_inchannels[i], momentum
                        =_BN_MOMENTUM), nn.Upsample(scale_factor=2 ** (j -
                        i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3, momentum=
                                _BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3, momentum=
                                _BN_MOMENTUM), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1001, in_chans=3, drop_rate=0.0,
        global_pool='avg'):
        super(InceptionResnetV2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        self.conv2d_1a = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
        self.block8 = Block8(no_relu=True)
        self.conv2d_7b = BasicConv2d(2080, self.num_features, kernel_size=1,
            stride=1)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classif = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), num_classes)

    def get_classifier(self):
        return self.classif

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classif = nn.Linear(num_features, num_classes)
        else:
            self.classif = nn.Identity()

    def forward_features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classif(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.'
            , stacklevel=2)
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


class InceptionV3Aux(nn.Module):
    """InceptionV3 with AuxLogits
    """

    def __init__(self, inception_blocks=None, num_classes=1000, in_chans=3,
        drop_rate=0.0, global_pool='avg'):
        super(InceptionV3Aux, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB,
                InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.Conv2d_1a_3x3 = conv_block(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.num_features = 2048
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux = self.AuxLogits(x) if self.training else None
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x, aux

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.fc = nn.Linear(self.num_features * self.global_pool.
                feat_mult(), num_classes)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x, aux = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x, aux


class InceptionV3(nn.Module):
    """Inception-V3 with no AuxLogits
    FIXME two class defs are redundant, but less screwing around with torchsript fussyness and inconsistent returns
    """

    def __init__(self, inception_blocks=None, num_classes=1000, in_chans=3,
        drop_rate=0.0, global_pool='avg'):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB,
                InceptionC, InceptionD, InceptionE]
        assert len(inception_blocks) >= 6
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        self.Conv2d_1a_3x3 = conv_block(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.num_features = 2048
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.fc = nn.Linear(self.num_features * self.global_pool.
                feat_mult(), num_classes)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1
            )

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(
            0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=
            (3, 0))
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride
            =1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3),
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(384, 96, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1,
            padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1,
            padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7),
            stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7,
            1), stride=1, padding=(3, 0)), BasicConv2d(224, 256,
            kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1),
            stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3,
            stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1,
            padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1,
            padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001, in_chans=3, drop_rate=0.0,
        global_pool='avg'):
        super(InceptionV4, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        self.features = nn.Sequential(BasicConv2d(in_chans, 32, kernel_size
            =3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(
            ), Inception_A(), Inception_A(), Reduction_A(), Inception_B(),
            Inception_B(), Inception_B(), Inception_B(), Inception_B(),
            Inception_B(), Inception_B(), Reduction_B(), Inception_C(),
            Inception_C(), Inception_C())
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.last_linear = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), num_classes)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.last_linear = nn.Linear(num_features, num_classes)
        else:
            self.last_linear = nn.Identity()

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


def swish(x, inplace: bool=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def mish(x, inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """

    def __init__(self, inplace: bool=False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class Sigmoid(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


class Tanh(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool=False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool=False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


@torch.jit.script
def swish_jit(x, inplace: bool=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


class SwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


@torch.jit.script
def mish_jit(x, _inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class MishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


class HardSigmoidJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool=False):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


class HardSwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


class SwishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp *
        x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


class MishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def hard_sigmoid_jit_fwd(x, inplace: bool=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


@torch.jit.script
def hard_sigmoid_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.0) & (x <= 3.0)) / 6.0
    return grad_output * m


class HardSigmoidJitAutoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(x, grad_output)


class HardSigmoidMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidJitAutoFn.apply(x)


@torch.jit.script
def hard_swish_jit_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


@torch.jit.script
def hard_swish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.0)
    m = torch.where((x >= -3.0) & (x <= 3.0), x / 3.0 + 0.5, m)
    return grad_output * m


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(x, grad_output)


class HardSwishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishJitAutoFn.apply(x)


@torch.jit.script
def hard_mish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.0)
    m = torch.where((x >= -2.0) & (x <= 0.0), x + 1.0, m)
    return grad_output * m


@torch.jit.script
def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJitAutoFn(torch.autograd.Function):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_jit_bwd(x, grad_output)


class HardMishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishJitAutoFn.apply(x)


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten
        if pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                assert False, 'Invalid pool type: %s' % pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.
            output_size) + ', pool_type=' + self.pool_type + ')'


class Downsample(nn.Module):

    def __init__(self, channels=None, filt_size=3, stride=2):
        super(Downsample, self).__init__()
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        assert self.filt_size == 3
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt = filt[:, (None)] * filt[(None), :]
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[(None), (None), :, :].repeat((
            self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0,
            groups=input.shape[1])


class BlurPool2d(nn.Module):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    filt: Dict[str, torch.Tensor]

    def __init__(self, channels, filt_size=3, stride=2) ->None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        pad_size = [get_padding(filt_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.
            filt_size - 1)).coeffs)
        self.filt = {}

    def _create_filter(self, like: torch.Tensor):
        blur_filter = (self._coeffs[:, (None)] * self._coeffs[(None), :]).to(
            dtype=like.dtype, device=like.device)
        return blur_filter[(None), (None), :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn):
        self.filt = {}
        super(BlurPool2d, self)._apply(fn)

    def forward(self, input_tensor: torch.Tensor) ->torch.Tensor:
        C = input_tensor.shape[1]
        blur_filt = self.filt.get(str(input_tensor.device), self.
            _create_filter(input_tensor))
        return F.conv2d(self.padding(input_tensor), blur_filt, stride=self.
            stride, groups=C)


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_avg = self.fc2(self.act(self.fc1(x_avg)))
        x_max = self.fc2(self.act(self.fc1(x_max)))
        x_attn = x_avg + x_max
        return x * x_attn.sigmoid()


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = torch.cat([x_avg, x_max], dim=1)
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """

    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvBnAct(1, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = 0.5 * x_avg + 0.5 * x_max
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class CbamModule(nn.Module):

    def __init__(self, channels, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """

    def __init__(self, channels, reduction=16):
        super(LightChannelAttn, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = 0.5 * self.avg_pool(x) + 0.5 * self.max_pool(x)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * x_attn.sigmoid()


class LightCbamModule(nn.Module):

    def __init__(self, channels, spatial_kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


tup_pair = _ntuple(2)


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: List[int], s: List[int], d: List[int]=(1, 1), value: float=0
    ):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw,
        k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - 
            pad_h // 2], value=value)
    return x


def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor]=None,
    stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0),
    dilation: Tuple[int, int]=(1, 1), groups: int=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tup_pair(kernel_size)
        self.stride = tup_pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding,
            kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = tup_pair(padding_val)
        self.dilation = tup_pair(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (self.out_channels, self.in_channels // self.groups
            ) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts,
            weight_num_param))
        if bias:
            self.bias_shape = self.out_channels,
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts,
                self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(partial(nn.init.
            kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.
            weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.init.uniform_,
                a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self
            .groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(x, weight, bias, stride=self.stride, padding=
                self.padding, dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(x, weight, bias, stride=self.stride, padding=
                self.padding, dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.
            shape[-2], out.shape[-1])
        return out


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def drop_block_2d(x, drop_prob: float=0.1, block_size: int=7, gamma_scale:
    float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool
    =False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((
        W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H)
        .to(x.device))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (
        clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) &
        (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device
            )
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = (2 - gamma - valid_block + uniform_noise >= 1).to(dtype=x.
        dtype)
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size,
        stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device
            ) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.
            float32).sum().add(1e-07)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor, drop_prob: float=0.1, block_size:
    int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=
    False, batchwise: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((
        W - block_size + 1) * (H - block_size + 1))
    if batchwise:
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device
            ) < gamma
    else:
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(block_mask.to(x.dtype), kernel_size=
        clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device
            ) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.
            float32).sum().add(1e-07)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0,
        with_noise=False, inplace=False, batchwise=False, fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size,
                self.gamma_scale, self.with_noise, self.inplace, self.batchwise
                )
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.
                gamma_scale, self.with_noise, self.inplace, self.batchwise)


class DropPath(nn.ModuleDict):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class EcaModule(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(
            kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)


class CecaModule(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without significantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(CecaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0,
            bias=False)
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        y = self.avg_pool(x)
        y = F.pad(y.view(x.shape[0], 1, -1), (self.padding, self.padding),
            mode='circular')
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)


class EvoNormBatch2d(nn.Module):

    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=
        1e-05, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        param_shape = 1, num_features, 1, 1
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_var.copy_(self.momentum * var.detach() + (1 - self
                .momentum) * self.running_var)
        else:
            var = self.running_var
        if self.apply_act:
            v = self.v.to(dtype=x_type)
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) +
                self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):

    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-05,
        drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act
        self.groups = groups
        self.eps = eps
        param_shape = 1, num_features, 1, 1
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = (x * self.v).sigmoid().reshape(B, self.groups, -1)
            x = x.reshape(B, self.groups, -1)
            x = n / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps
                ).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias


class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        apply_act=True, act_layer='leaky_relu', act_param=0.01, drop_block=None
        ):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity')
                self.act_name = act_layer
            elif isinstance(act_layer, nn.ELU):
                self.act_name = 'elu'
            elif isinstance(act_layer, nn.LeakyReLU):
                self.act_name = 'leaky_relu'
            else:
                assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        output = inplace_abn(x, self.weight, self.bias, self.running_mean,
            self.running_var, self.training, self.momentum, self.eps, self.
            act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - ih % self.stride[0], 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - iw % self.stride[1], 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = pl, pr, pt, pb
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1],
            self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, apply_act=True, act_layer=nn.ReLU,
        inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine, track_running_stats=
            track_running_stats)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer(inplace=inplace)
        else:
            self.act = None

    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm2d forward function
        """
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.
                        num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        x = F.batch_norm(x, self.running_mean, self.running_var, self.
            weight, self.bias, self.training or not self.
            track_running_stats, exponential_average_factor, self.eps)
        return x

    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct2d, self).forward(x)

    def forward(self, x):
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GroupNormAct(nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
        apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=
            eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer(inplace=inplace)
        else:
            self.act = None

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        if self.act is not None:
            x = self.act(x)
        return x


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], padding:
    List[int]=(0, 0), ceil_mode: bool=False, count_include_pad: bool=True):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode,
        count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=
        False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0),
            ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(x, self.kernel_size, self.stride, self.
            padding, self.ceil_mode, self.count_include_pad)


def max_pool2d_same(x, kernel_size: List[int], stride: List[int], padding:
    List[int]=(0, 0), dilation: List[int]=(1, 1), ceil_mode: bool=False):
    x = pad_same(x, kernel_size, stride, value=-float('inf'))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1,
        ceil_mode=False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0),
            dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        return max_pool2d_same(x, self.kernel_size, self.stride, self.
            padding, self.dilation, self.ceil_mode)


def tanh(x, inplace: bool=False):
    return x.tanh_() if inplace else x.tanh()


_ACT_FN_DEFAULT = dict(swish=swish, mish=mish, relu=F.relu, relu6=F.relu6,
    leaky_relu=F.leaky_relu, elu=F.elu, prelu=F.prelu, celu=F.celu, selu=F.
    selu, gelu=F.gelu, sigmoid=sigmoid, tanh=tanh, hard_sigmoid=
    hard_sigmoid, hard_swish=hard_swish, hard_mish=hard_mish)


_ACT_FN_JIT = dict(swish=swish_jit, mish=mish_jit, hard_sigmoid=
    hard_sigmoid_jit, hard_swish=hard_swish_jit, hard_mish=hard_mish_jit)


_NO_JIT = False


def is_no_jit():
    return _NO_JIT


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


def swish_me(x, inplace=False):
    return SwishJitAutoFn.apply(x)


def hard_swish_me(x, inplace=False):
    return HardSwishJitAutoFn.apply(x)


def hard_sigmoid_me(x, inplace: bool=False):
    return HardSigmoidJitAutoFn.apply(x)


def hard_mish_me(x, inplace: bool=False):
    return HardMishJitAutoFn.apply(x)


_ACT_FN_ME = dict(swish=swish_me, mish=mish_me, hard_sigmoid=
    hard_sigmoid_me, hard_swish=hard_swish_me, hard_mish=hard_mish_me)


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    if not is_no_jit():
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU,
        min_channels=8, reduction_channels=None, gate_fn='sigmoid'):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = reduction_channels or max(channels //
            reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1,
            padding=0, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1,
            padding=0, bias=True)
        self.gate_fn = get_act_fn(gate_fn)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate_fn(x_se)


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channel, gate_fn='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.gate_fn = get_act_fn(gate_fn)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc(x_se)
        return x * self.gate_fn(x_se, inplace=True)


class SelectiveKernelAttn(nn.Module):

    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=
        nn.ReLU, norm_layer=nn.BatchNorm2d):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1,
            bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths,
            kernel_size=1, bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = torch.sum(x, dim=1)
        x = self.pool(x)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=
        1, dilation=1, groups=1, attn_reduction=16, min_attn_channels=32,
        keep_3x3=True, split_input=False, drop_block=None, act_layer=nn.
        ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels (int):  module input (feature) channel count
            out_channels (int):  module output (feature) channel count
            kernel_size (int, list): kernel size for each convolution branch
            stride (int): stride for convolutions
            dilation (int): dilation for module as a whole, impacts dilation of each branch
            groups (int): number of groups for each branch
            attn_reduction (int, float): reduction factor for attention features
            min_attn_channels (int): minimum attention feature channels
            keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            drop_block (nn.Module): drop block module
            act_layer (nn.Module): activation layer to use
            norm_layer (nn.Module): batchnorm/norm layer to use
        """
        super(SelectiveKernelConv, self).__init__()
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [(dilation * (k - 1) // 2) for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)
        conv_kwargs = dict(stride=stride, groups=groups, drop_block=
            drop_block, act_layer=act_layer, norm_layer=norm_layer,
            aa_layer=aa_layer)
        self.paths = nn.ModuleList([ConvBnAct(in_channels, out_channels,
            kernel_size=k, dilation=d, **conv_kwargs) for k, d in zip(
            kernel_size, dilation)])
        attn_channels = max(int(out_channels / attn_reduction),
            min_attn_channels)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths,
            attn_channels)
        self.drop_block = drop_block

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = torch.sum(x, dim=1)
        return x


class SeparableConvBnAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, padding='', bias=False, channel_multiplier=1.0,
        pw_kernel_size=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        act_layer=nn.ReLU, apply_act=True, drop_block=None):
        super(SeparableConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv_dw = create_conv2d(in_channels, int(in_channels *
            channel_multiplier), kernel_size, stride=stride, dilation=
            dilation, padding=padding, depthwise=True)
        self.conv_pw = create_conv2d(int(in_channels * channel_multiplier),
            out_channels, pw_kernel_size, padding=padding, bias=bias)
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer,
            act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act,
            drop_block=drop_block, **norm_act_args)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, padding='', bias=False, channel_multiplier=1.0,
        pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(in_channels, int(in_channels *
            channel_multiplier), kernel_size, stride=stride, dilation=
            dilation, padding=padding, depthwise=True)
        self.conv_pw = create_conv2d(int(in_channels * channel_multiplier),
            out_channels, pw_kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class RadixSoftmax(nn.Module):

    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttnConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False, radix=2,
        reduction_factor=4, act_layer=nn.ReLU, norm_layer=None, drop_block=
        None, **kwargs):
        super(SplitAttnConv2d, self).__init__()
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        attn_chs = max(in_channels * radix // reduction_factor, 32)
        self.conv = nn.Conv2d(in_channels, mid_chs, kernel_size, stride,
            padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer is not None else None
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer is not None else None
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)
        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = F.adaptive_avg_pool2d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        if self.bn1 is not None:
            x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))
                ).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine,
            track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([nn.BatchNorm2d(num_features, eps,
            momentum, affine, track_running_stats) for _ in range(
            num_splits - 1)])

    def forward(self, input: torch.Tensor):
        if self.training:
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0
                ] == split_size * self.num_splits, 'batch size must be evenly divisible by num_splits'
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)


class TestTimePoolHead(nn.Module):

    def __init__(self, base, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.base = base
        self.original_pool = original_pool
        base_fc = self.base.get_classifier()
        if isinstance(base_fc, nn.Conv2d):
            self.fc = base_fc
        else:
            self.fc = nn.Conv2d(self.base.num_features, self.base.
                num_classes, kernel_size=1, bias=True)
            self.fc.weight.data.copy_(base_fc.weight.data.view(self.fc.
                weight.size()))
            self.fc.bias.data.copy_(base_fc.bias.data.view(self.fc.bias.size())
                )
        self.base.reset_classifier(0)

    def forward(self, x):
        x = self.base.forward_features(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=
        16, num_features=1280, head_bias=True, channel_multiplier=1.0,
        pad_type='', act_layer=nn.ReLU, drop_rate=0.0, drop_path_rate=0.0,
        se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        global_pool='avg'):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size
        builder = EfficientNetBuilder(channel_multiplier, 8, None, 32,
            pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs,
            drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1,
            padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.classifier = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), self.num_classes)
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.
            classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4),
        feature_location='bottleneck', in_chans=3, stem_size=16,
        channel_multiplier=1.0, output_stride=32, pad_type='', act_layer=nn
        .ReLU, drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(MobileNetV3Features, self).__init__()
        norm_kwargs = norm_kwargs or {}
        num_stages = max(out_indices) + 1
        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size
        builder = EfficientNetBuilder(channel_multiplier, 8, None,
            output_stride, pad_type, act_layer, se_kwargs, norm_layer,
            norm_kwargs, drop_path_rate, feature_location=feature_location,
            verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self._feature_info = builder.features
        self._stage_to_feature_idx = {v['stage_idx']: fi for fi, v in self.
            _feature_info.items() if fi in self.out_indices}
        self._in_chs = builder.in_chs
        efficientnet_init_weights(self)
        if _DEBUG:
            for k, v in self._feature_info.items():
                None
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = [dict(name=self._feature_info[idx]['module'], type=self
                ._feature_info[idx]['hook_type']) for idx in out_indices]
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def feature_channels(self, idx=None):
        """ Feature Channel Shortcut
        Returns feature channel count for each output index if idx == None. If idx is an integer, will
        return feature channel count for that feature block index (independent of out_indices setting).
        """
        if isinstance(idx, int):
            return self._feature_info[idx]['num_chs']
        return [self._feature_info[i]['num_chs'] for i in self.out_indices]

    def feature_info(self, idx=None):
        """ Feature Channel Shortcut
        Returns feature channel count for each output index if idx == None. If idx is an integer, will
        return feature channel count for that feature block index (independent of out_indices setting).
        """
        if isinstance(idx, int):
            return self._feature_info[idx]
        return [self._feature_info[i] for i in self.out_indices]

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i in self._stage_to_feature_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            return self.feature_hooks.get_output(x.device)


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_size, num_channels=42):
        super(CellStem0, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_size, self.
            num_channels, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_channels,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.
            num_channels, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_size, self.
            num_channels, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_size, self.
            num_channels, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_size, self.
            num_channels, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.
            num_channels, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_size, num_channels):
        super(CellStem1, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_channels,
            self.num_channels, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_channels,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_size, self.
            num_channels // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_size, self.
            num_channels // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_channels, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.
            num_channels, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_channels, self.
            num_channels, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(self.num_channels, self.
            num_channels, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(self.num_channels, self.
            num_channels, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.
            num_channels, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels,
            kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1000, in_chans=1, stem_size=96,
        num_features=4032, channel_multiplier=2, drop_rate=0.0, global_pool
        ='avg'):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_size = stem_size
        self.num_features = num_features
        self.channel_multiplier = channel_multiplier
        self.drop_rate = drop_rate
        channels = self.num_features // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=in_chans,
            out_channels=self.stem_size, kernel_size=3, padding=0, stride=2,
            bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_size, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_size, num_channels=channels //
            channel_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_size, num_channels=channels //
            channel_multiplier)
        self.cell_0 = FirstCell(in_channels_left=channels,
            out_channels_left=channels // 2, in_channels_right=2 * channels,
            out_channels_right=channels)
        self.cell_1 = NormalCell(in_channels_left=2 * channels,
            out_channels_left=channels, in_channels_right=6 * channels,
            out_channels_right=channels)
        self.cell_2 = NormalCell(in_channels_left=6 * channels,
            out_channels_left=channels, in_channels_right=6 * channels,
            out_channels_right=channels)
        self.cell_3 = NormalCell(in_channels_left=6 * channels,
            out_channels_left=channels, in_channels_right=6 * channels,
            out_channels_right=channels)
        self.cell_4 = NormalCell(in_channels_left=6 * channels,
            out_channels_left=channels, in_channels_right=6 * channels,
            out_channels_right=channels)
        self.cell_5 = NormalCell(in_channels_left=6 * channels,
            out_channels_left=channels, in_channels_right=6 * channels,
            out_channels_right=channels)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 *
            channels, out_channels_left=2 * channels, in_channels_right=6 *
            channels, out_channels_right=2 * channels)
        self.cell_6 = FirstCell(in_channels_left=6 * channels,
            out_channels_left=channels, in_channels_right=8 * channels,
            out_channels_right=2 * channels)
        self.cell_7 = NormalCell(in_channels_left=8 * channels,
            out_channels_left=2 * channels, in_channels_right=12 * channels,
            out_channels_right=2 * channels)
        self.cell_8 = NormalCell(in_channels_left=12 * channels,
            out_channels_left=2 * channels, in_channels_right=12 * channels,
            out_channels_right=2 * channels)
        self.cell_9 = NormalCell(in_channels_left=12 * channels,
            out_channels_left=2 * channels, in_channels_right=12 * channels,
            out_channels_right=2 * channels)
        self.cell_10 = NormalCell(in_channels_left=12 * channels,
            out_channels_left=2 * channels, in_channels_right=12 * channels,
            out_channels_right=2 * channels)
        self.cell_11 = NormalCell(in_channels_left=12 * channels,
            out_channels_left=2 * channels, in_channels_right=12 * channels,
            out_channels_right=2 * channels)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            channels, out_channels_left=4 * channels, in_channels_right=12 *
            channels, out_channels_right=4 * channels)
        self.cell_12 = FirstCell(in_channels_left=12 * channels,
            out_channels_left=2 * channels, in_channels_right=16 * channels,
            out_channels_right=4 * channels)
        self.cell_13 = NormalCell(in_channels_left=16 * channels,
            out_channels_left=4 * channels, in_channels_right=24 * channels,
            out_channels_right=4 * channels)
        self.cell_14 = NormalCell(in_channels_left=24 * channels,
            out_channels_left=4 * channels, in_channels_right=24 * channels,
            out_channels_right=4 * channels)
        self.cell_15 = NormalCell(in_channels_left=24 * channels,
            out_channels_left=4 * channels, in_channels_right=24 * channels,
            out_channels_right=4 * channels)
        self.cell_16 = NormalCell(in_channels_left=24 * channels,
            out_channels_left=4 * channels, in_channels_right=24 * channels,
            out_channels_right=4 * channels)
        self.cell_17 = NormalCell(in_channels_left=24 * channels,
            out_channels_left=4 * channels, in_channels_right=24 * channels,
            out_channels_right=4 * channels)
        self.relu = nn.ReLU()
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.last_linear = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), num_classes)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.last_linear = nn.Linear(num_features, num_classes)
        else:
            self.last_linear = nn.Identity()

    def forward_features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        x = self.relu(x_cell_17)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad is not None:
            x = self.zero_pad(x)
            x = self.pool(x)
            x = x[:, :, 1:, 1:]
        else:
            x = self.pool(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
        dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            kernel_size=dw_kernel_size, stride=dw_stride, padding=
            dw_padding, groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
            kernel_size, dw_stride=stride, dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
            kernel_size, dw_stride=1, dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu_1(x)
        if self.zero_pad is not None:
            x = self.zero_pad(x)
            x = self.separable_1(x)
            x = x[:, :, 1:, 1:].contiguous()
        else:
            x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReluConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1,
            stride=2, count_include_pad=False)), ('conv', nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1, bias=False))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((0, 1,
            0, 1))), ('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False)), ('conv', nn.Conv2d(in_channels, 
            out_channels // 2, kernel_size=1, bias=False))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
            x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class Cell(CellBase):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right, is_reduction=False, zero_pad
        =False, match_prev_layer_dimensions=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left,
                out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left,
                out_channels_left, kernel_size=1)
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
            kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=7, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=5, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=3, stride=stride, zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                out_channels_right, kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1001, in_chans=3, drop_rate=0.5,
        global_pool='avg'):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.num_features = 4320
        self.drop_rate = drop_rate
        self.conv_0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_chans, 96, kernel_size=3, stride=2, bias=False)), ('bn', nn.
            BatchNorm2d(96, eps=0.001))]))
        self.cell_stem_0 = CellStem0(in_channels_left=96, out_channels_left
            =54, in_channels_right=96, out_channels_right=54)
        self.cell_stem_1 = Cell(in_channels_left=96, out_channels_left=108,
            in_channels_right=270, out_channels_right=108,
            match_prev_layer_dimensions=True, is_reduction=True)
        self.cell_0 = Cell(in_channels_left=270, out_channels_left=216,
            in_channels_right=540, out_channels_right=216,
            match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=540, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_2 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_3 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_4 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=1080, out_channels_right=432, is_reduction=
            True, zero_pad=True)
        self.cell_5 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432,
            match_prev_layer_dimensions=True)
        self.cell_6 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_7 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_8 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=2160, out_channels_right=864, is_reduction=True)
        self.cell_9 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864,
            match_prev_layer_dimensions=True)
        self.cell_10 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.cell_11 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.relu = nn.ReLU()
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.last_linear = nn.Linear(self.num_features * self.global_pool.
            feat_mult(), num_classes)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.last_linear = nn.Linear(num_features, num_classes)
        else:
            self.last_linear = nn.Identity()

    def forward_features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.relu(x_cell_11)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, bottleneck_ratio=1,
        group_width=1, se_ratio=0.25, dilation=1, first_dilation=None,
        downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
        aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width
        first_dilation = first_dilation or dilation
        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=
            aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvBnAct(bottleneck_chs, bottleneck_chs, kernel_size=
            3, stride=stride, dilation=first_dilation, groups=groups, **cargs)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, reduction_channels=se_channels)
        else:
            self.se = None
        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


def downsample_conv(in_channels, out_channels, kernel_size, stride=1,
    dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = first_dilation or dilation if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)
    return nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size,
        stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, depth, block_fn,
        bottle_ratio, group_width, se_ratio):
        super(RegStage, self).__init__()
        block_kwargs = {}
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            if block_in_chs != out_chs or block_stride != 1:
                proj_block = downsample_conv(block_in_chs, out_chs, 1, stride)
            else:
                proj_block = None
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fn(block_in_chs, out_chs,
                block_stride, bottle_ratio, group_width, se_ratio,
                downsample=proj_block, **block_kwargs))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ClassifierHead(nn.Module):
    """Head."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.0):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if num_classes > 0:
            self.fc = nn.Linear(in_chs, num_classes, bias=True)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(
        bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths,
        bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(
        width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg',
        drop_rate=0.0, zero_init_last_bn=True):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        stem_width = cfg['stem_width']
        self.stem = ConvBnAct(in_chans, stem_width, 3, stride=2)
        block_fn = Bottleneck
        prev_width = stem_width
        stage_params = self._get_stage_params(cfg)
        se_ratio = cfg['se_ratio']
        for i, (d, w, s, br, gw) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), RegStage(prev_width, w, s,
                d, block_fn, br, gw, se_ratio))
            prev_width = w
        self.num_features = prev_width
        self.head = ClassifierHead(in_chs=prev_width, num_classes=
            num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, stride=2):
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = [stride for _ in range(num_stages)]
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths,
            stage_bottle_ratios, stage_groups)
        stage_params = list(zip(stage_depths, stage_widths, stage_strides,
            stage_bottle_ratios, stage_groups))
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes,
            pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=
        None, act_layer=nn.ReLU, norm_layer=None, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias
            =False)
        self.bn1 = norm_layer(width * scale)
        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=
                stride, padding=first_dilation, dilation=first_dilation,
                groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None
        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1,
            bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None
        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1,
        cardinality=1, base_width=64, avd=False, avd_first=False, is_first=
        False, reduce_first=1, dilation=1, first_dilation=None, act_layer=
        nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
        drop_block=None, drop_path=None):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1
        assert attn_layer is None
        assert aa_layer is None
        assert drop_path is None
        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix
        self.drop_block = drop_block
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False
            )
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1
            ) if avd_stride > 0 and avd_first else None
        if self.radix >= 1:
            self.conv2 = SplitAttnConv2d(group_width, group_width,
                kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix,
                norm_layer=norm_layer, drop_block=drop_block)
            self.bn2 = None
            self.act2 = None
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=first_dilation, dilation=
                first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1
            ) if avd_stride > 0 and not avd_first else None
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias
            =False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.act1(out)
        if self.avd_first is not None:
            out = self.avd_first(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
            if self.drop_block is not None:
                out = self.drop_block(out)
            out = self.act2(out)
        if self.avd_last is not None:
            out = self.avd_last(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act3(out)
        return out


def create_attn(attn_type, channels, **kwargs):
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            else:
                assert False, 'Invalid attn module (%s)' % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        cardinality=1, base_width=64, reduce_first=1, dilation=1,
        first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
        attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=3,
            stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes
            ) if stride == 2 and use_aa else None
        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.se = create_attn(attn_layer, outplanes)
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        cardinality=1, base_width=64, reduce_first=1, dilation=1,
        first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
        attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=
            False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(first_planes, width, kernel_size=3, stride=1 if
            use_aa else stride, padding=first_dilation, dilation=
            first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width) if stride == 2 and use_aa else None
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = create_attn(attn_layer, outplanes)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)
        return x


def downsample_avg(in_channels, out_channels, kernel_size, stride=1,
    dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = (AvgPool2dSame if avg_stride == 1 and dilation > 1 else
            nn.AvgPool2d)
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad
            =False)
    return nn.Sequential(*[pool, nn.Conv2d(in_channels, out_channels, 1,
        stride=1, padding=0, bias=False), norm_layer(out_channels)])


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'deep_tiered_narrow' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
        cardinality=1, base_width=64, stem_width=64, stem_type='',
        block_reduce_first=1, down_kernel_size=1, avg_down=False,
        output_stride=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
        aa_layer=None, drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=
        0.0, global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        self.num_classes = num_classes
        deep_stem = 'deep' in stem_type
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        super(ResNet, self).__init__()
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (
                    stem_width // 4)
            self.conv1 = nn.Sequential(*[nn.Conv2d(in_chans, stem_chs_1, 3,
                stride=2, padding=1, bias=False), norm_layer(stem_chs_1),
                act_layer(inplace=True), nn.Conv2d(stem_chs_1, stem_chs_2, 
                3, stride=1, padding=1, bias=False), norm_layer(stem_chs_2),
                act_layer(inplace=True), nn.Conv2d(stem_chs_2, self.
                inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, self.inplanes, kernel_size=7,
                stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = act_layer(inplace=True)
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=3,
                stride=1, padding=1), aa_layer(channels=self.inplanes,
                stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        dp = DropPath(drop_path_rate) if drop_path_rate else None
        db_3 = DropBlock2d(drop_block_rate, 7, 0.25
            ) if drop_block_rate else None
        db_4 = DropBlock2d(drop_block_rate, 7, 1.0
            ) if drop_block_rate else None
        channels, strides, dilations = [64, 128, 256, 512], [1, 2, 2, 2], [1
            ] * 4
        if output_stride == 16:
            strides[3] = 1
            dilations[3] = 2
        elif output_stride == 8:
            strides[2:4] = [1, 1]
            dilations[2:4] = [2, 4]
        else:
            assert output_stride == 32
        layer_args = list(zip(channels, layers, strides, dilations))
        layer_kwargs = dict(reduce_first=block_reduce_first, act_layer=
            act_layer, norm_layer=norm_layer, aa_layer=aa_layer, avg_down=
            avg_down, down_kernel_size=down_kernel_size, drop_path=dp, **
            block_args)
        self.layer1 = self._make_layer(block, *layer_args[0], **layer_kwargs)
        self.layer2 = self._make_layer(block, *layer_args[1], **layer_kwargs)
        self.layer3 = self._make_layer(block, *layer_args[2], drop_block=
            db_3, **layer_kwargs)
        self.layer4 = self._make_layer(block, *layer_args[3], drop_block=
            db_4, **layer_kwargs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        reduce_first=1, avg_down=False, down_kernel_size=1, **kwargs):
        downsample = None
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_args = dict(in_channels=self.inplanes, out_channels=
                planes * block.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=
                first_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**downsample_args
                ) if avg_down else downsample_conv(**downsample_args)
        block_kwargs = dict(cardinality=self.cardinality, base_width=self.
            base_width, reduce_first=reduce_first, dilation=dilation, **kwargs)
        layers = [block(self.inplanes, planes, stride, downsample,
            first_dilation=first_dilation, **block_kwargs)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, **block_kwargs) for _ in
            range(1, blocks)]
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.fc = nn.Linear(num_features, num_classes)
        else:
            self.fc = nn.Identity()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


class SequentialList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


def conv_bn(in_chs, out_chs, k=3, stride=1, padding=None, dilation=1):
    if padding is None:
        padding = (stride - 1 + dilation * (k - 1)) // 2
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, k, stride, padding=
        padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True))


class SelecSLSBlock(nn.Module):

    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride,
        dilation=1):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]
        self.conv1 = conv_bn(in_chs, mid_chs, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(mid_chs, mid_chs, 1)
        self.conv3 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_bn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_bn(2 * mid_chs + (0 if is_first else skip_chs),
            out_chs, 1)

    def forward(self, x: List[torch.Tensor]) ->List[torch.Tensor]:
        assert isinstance(x, list)
        assert len(x) in [1, 2]
        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLS(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, cfg, num_classes=1000, in_chans=3, drop_rate=0.0,
        global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(SelecSLS, self).__init__()
        self.stem = conv_bn(in_chans, 32, stride=2)
        self.features = SequentialList(*[cfg['block'](*block_args) for
            block_args in cfg['features']])
        self.head = nn.Sequential(*[conv_bn(*conv_args) for conv_args in
            cfg['head']])
        self.num_features = cfg['num_features']
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.fc = nn.Linear(num_features, num_classes)
        else:
            self.fc = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
        downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
            stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
            groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2,
        in_chans=3, inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, inplanes,
                kernel_size=7, stride=2, padding=3, bias=False)), ('bn1',
                nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.last_linear = nn.Linear(self.num_features, num_classes)
        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, groups, reduction, stride,
            downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.avg_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.avg_pool.feat_mult()
            self.last_linear = nn.Linear(num_features, num_classes)
        else:
            self.last_linear = nn.Identity()

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x).flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1,
        dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.
        BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None,
        drop_path=None):
        super(SelectiveKernelBasic, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer,
            norm_layer=norm_layer, aa_layer=aa_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = SelectiveKernelConv(inplanes, first_planes, stride=
            stride, dilation=first_dilation, **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv2 = ConvBnAct(first_planes, outplanes, kernel_size=3,
            dilation=dilation, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1,
        dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.
        BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None,
        drop_path=None):
        super(SelectiveKernelBottleneck, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer,
            norm_layer=norm_layer, aa_layer=aa_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **
            conv_kwargs)
        self.conv2 = SelectiveKernelConv(first_planes, width, stride=stride,
            dilation=first_dilation, groups=cardinality, **conv_kwargs, **
            sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class FastGlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0),
                x.size(1), 1, 1)

    def feat_mult(self):
        return 1


class FastSEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(FastSEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1,
            padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1,
            padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer=
    'leaky_relu', act_param=0.01):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=
        stride, padding=kernel_size // 2, groups=groups, bias=False),
        InplaceAbn(nf, act_layer=act_layer, act_param=act_param))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=
        True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=
                0.001)
        elif aa_layer is None:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=
                0.001)
        else:
            self.conv1 = nn.Sequential(conv2d_iabn(inplanes, planes, stride
                =1, act_param=0.001), aa_layer(channels=planes, filt_size=3,
                stride=2))
        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer='identity'
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = FastSEModule(planes * self.expansion, reduce_layer_planes
            ) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=
        True, act_layer='leaky_relu', aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(inplanes, planes, kernel_size=1, stride=1,
            act_layer=act_layer, act_param=0.001)
        if stride == 1:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=
                1, act_layer=act_layer, act_param=0.001)
        elif aa_layer is None:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=
                2, act_layer=act_layer, act_param=0.001)
        else:
            self.conv2 = nn.Sequential(conv2d_iabn(planes, planes,
                kernel_size=3, stride=1, act_layer=act_layer, act_param=
                0.001), aa_layer(channels=planes, filt_size=3, stride=2))
        self.conv3 = conv2d_iabn(planes, planes * self.expansion,
            kernel_size=1, stride=1, act_layer='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = FastSEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + residual
        out = self.relu(out)
        return out


class TResNet(nn.Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=
        1.0, no_aa_jit=False, global_pool='avg', drop_rate=0.0):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TResNet, self).__init__()
        space_to_depth = SpaceToDepthModule()
        aa_layer = partial(AntiAliasDownsampleLayer, no_jit=no_aa_jit)
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_chans * 16, self.planes, stride=1, kernel_size=3
            )
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0],
            stride=1, use_se=True, aa_layer=aa_layer)
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1],
            stride=2, use_se=True, aa_layer=aa_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2],
            stride=2, use_se=True, aa_layer=aa_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3],
            stride=2, use_se=False, aa_layer=aa_layer)
        self.body = nn.Sequential(OrderedDict([('SpaceToDepth',
            space_to_depth), ('conv1', conv1), ('layer1', layer1), (
            'layer2', layer2), ('layer3', layer3), ('layer4', layer4)]))
        self.num_features = self.planes * 8 * Bottleneck.expansion
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool,
            flatten=True)
        self.head = nn.Sequential(OrderedDict([('fc', nn.Linear(self.
            num_features * self.global_pool.feat_mult(), num_classes))]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1
                    ].weight))
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1
                    ].weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True,
        aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2,
                    ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=1, act_layer='identity')]
            downsample = nn.Sequential(*layers)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se,
                aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool,
            flatten=True)
        self.num_classes = num_classes
        self.head = None
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.head = nn.Sequential(OrderedDict([('fc', nn.Linear(
                num_features, num_classes))]))
        else:
            self.head = nn.Sequential(OrderedDict([('fc', nn.Identity())]))

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.head(x)
        return x


class SequentialAppendList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]
        ) ->torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x


class OsaBlock(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, layer_per_block, residual=
        False, depthwise=False, attn='', norm_layer=BatchNormAct2d):
        super(OsaBlock, self).__init__()
        self.residual = residual
        self.depthwise = depthwise
        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvBnAct(next_in_chs, mid_chs, 1,
                norm_layer=norm_layer)
        else:
            self.conv_reduction = None
        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvBnAct(mid_chs, mid_chs, norm_layer=
                    norm_layer)
            else:
                conv = ConvBnAct(next_in_chs, mid_chs, 3, norm_layer=norm_layer
                    )
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvBnAct(next_in_chs, out_chs, norm_layer=
            norm_layer)
        if attn:
            self.attn = create_attn(attn, out_chs)
        else:
            self.attn = None

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + output[0]
        return x


class OsaStage(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, block_per_stage,
        layer_per_block, downsample=True, residual=True, depthwise=False,
        attn='ese', norm_layer=BatchNormAct2d):
        super(OsaStage, self).__init__()
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None
        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            blocks += [OsaBlock(in_chs if i == 0 else out_chs, mid_chs,
                out_chs, layer_per_block, residual=residual and i > 0,
                depthwise=depthwise, attn=attn if last_block else '',
                norm_layer=norm_layer)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x


class ClassifierHead(nn.Module):
    """Head."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.0):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if num_classes > 0:
            self.fc = nn.Linear(in_chs, num_classes, bias=True)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


class VovNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg',
        drop_rate=0.0, stem_stride=4, norm_layer=BatchNormAct2d):
        """ VovNet (v2)
        """
        super(VovNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert stem_stride in (4, 2)
        stem_chs = cfg['stem_chs']
        stage_conv_chs = cfg['stage_conv_chs']
        stage_out_chs = cfg['stage_out_chs']
        block_per_stage = cfg['block_per_stage']
        layer_per_block = cfg['layer_per_block']
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvBnAct if cfg['depthwise'] else ConvBnAct
        self.stem = nn.Sequential(*[ConvBnAct(in_chans, stem_chs[0], 3,
            stride=2, norm_layer=norm_layer), conv_type(stem_chs[0],
            stem_chs[1], 3, stride=1, norm_layer=norm_layer), conv_type(
            stem_chs[1], stem_chs[2], 3, stride=last_stem_stride,
            norm_layer=norm_layer)])
        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(residual=cfg['residual'], depthwise=cfg[
            'depthwise'], attn=cfg['attn'], norm_layer=norm_layer)
        stages = []
        for i in range(4):
            downsample = stem_stride == 2 or i > 0
            stages += [OsaStage(in_ch_list[i], stage_conv_chs[i],
                stage_out_chs[i], block_per_stage[i], layer_per_block,
                downsample=downsample, **stage_args)]
            self.num_features = stage_out_chs[i]
        self.stages = nn.Sequential(*stages)
        self.head = ClassifierHead(self.num_features, num_classes,
            pool_type=global_pool, drop_rate=drop_rate)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes,
            pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0,
        global_pool='avg'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = 2048
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,
            grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.num_features)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(
            ), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.fc = nn.Linear(num_features, num_classes)
        else:
            self.fc = nn.Identity()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rwightman_pytorch_image_models(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AdaptiveAvgMaxPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(AdaptiveCatAvgMaxPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(AvgPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(BatchNormAct2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Block(*[], **{'in_filters': 4, 'out_filters': 4, 'reps': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Block17(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_007(self):
        self._check(Block35(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_008(self):
        self._check(Block8(*[], **{}), [torch.rand([4, 2080, 64, 64])], {})

    def test_009(self):
        self._check(BlurPool2d(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(BnActConv2d(*[], **{'in_chs': 4, 'out_chs': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(BranchSeparablesStem(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(CatBnAct(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(CecaModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(ChannelAttn(*[], **{'channels': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_015(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(ClassifierHead(*[], **{'in_chs': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(Conv2dSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_018(self):
        self._check(DPN(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_019(self):
        self._check(DenseTransition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_020(self):
        self._check(DepthToSpace(*[], **{'block_size': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_021(self):
        self._check(DlaBasic(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_022(self):
        self._check(DlaBottle2neck(*[], **{'inplanes': 64, 'outplanes': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_023(self):
        self._check(DlaBottleneck(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(DropBlock2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_025(self):
        self._check(DropPath(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_026(self):
        self._check(EcaModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_027(self):
        self._check(EffectiveSEModule(*[], **{'channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_028(self):
        self._check(EvoNormBatch2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_029(self):
        self._check(FactorizedReduction(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_030(self):
        self._check(FastGlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_031(self):
        self._check(FastSEModule(*[], **{'channels': 4, 'reduction_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_032(self):
        self._check(GroupNormAct(*[], **{'num_groups': 1, 'num_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_033(self):
        self._check(HardMish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_034(self):
        self._check(HardMishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_035(self):
        self._check(HardMishMe(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_036(self):
        self._check(HardSigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_037(self):
        self._check(HardSigmoidJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_038(self):
        self._check(HardSigmoidMe(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_039(self):
        self._check(HardSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_040(self):
        self._check(HardSwishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_041(self):
        self._check(HardSwishMe(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_042(self):
        self._check(Inception_A(*[], **{}), [torch.rand([4, 384, 64, 64])], {})

    def test_043(self):
        self._check(Inception_B(*[], **{}), [torch.rand([4, 1024, 64, 64])], {})

    def test_044(self):
        self._check(Inception_C(*[], **{}), [torch.rand([4, 1536, 64, 64])], {})

    def test_045(self):
        self._check(InputBlock(*[], **{'num_init_features': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_046(self):
        self._check(LabelSmoothingCrossEntropy(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_047(self):
        self._check(LightChannelAttn(*[], **{'channels': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_048(self):
        self._check(MaxPool(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_049(self):
        self._check(MaxPoolPad(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_050(self):
        self._check(MedianPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_051(self):
        self._check(Mish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_052(self):
        self._check(MishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_053(self):
        self._check(MishMe(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_054(self):
        self._check(Mixed_3a(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_055(self):
        self._check(Mixed_4a(*[], **{}), [torch.rand([4, 160, 64, 64])], {})

    def test_056(self):
        self._check(Mixed_5a(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_057(self):
        self._check(Mixed_5b(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_058(self):
        self._check(Mixed_6a(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_059(self):
        self._check(Mixed_7a(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_060(self):
        self._check(RadixSoftmax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_061(self):
        self._check(Reduction_A(*[], **{}), [torch.rand([4, 384, 64, 64])], {})

    def test_062(self):
        self._check(Reduction_B(*[], **{}), [torch.rand([4, 1024, 64, 64])], {})

    def test_063(self):
        self._check(ReluConvBn(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_064(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_065(self):
        self._check(SEResNetBlock(*[], **{'inplanes': 4, 'planes': 4, 'groups': 1, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_066(self):
        self._check(SelectAdaptivePool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_067(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_068(self):
        self._check(SequentialList(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_069(self):
        self._check(Sigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_070(self):
        self._check(SoftTargetCrossEntropy(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_071(self):
        self._check(SpaceToDepth(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_072(self):
        self._check(SplitAttnConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_073(self):
        self._check(SplitBatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_074(self):
        self._check(SqueezeExcite(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_075(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_076(self):
        self._check(SwishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_077(self):
        self._check(SwishMe(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_078(self):
        self._check(Tanh(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_079(self):
        self._check(Xception(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

