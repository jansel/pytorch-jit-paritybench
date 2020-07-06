import sys
_module = sys.modules[__name__]
del sys
augmentations = _module
calculate_class_weight = _module
cityscapes_loader = _module
cityscapesscripts = _module
annotation = _module
cityscapesLabelTool = _module
evaluation = _module
evalInstanceLevelSemanticLabeling = _module
evalPixelLevelSemanticLabeling = _module
instance = _module
instances2dict = _module
setup = _module
helpers = _module
csHelpers = _module
labels = _module
preparation = _module
createTrainIdInstanceImgs = _module
createTrainIdLabelImgs = _module
json2instanceImg = _module
json2labelImg = _module
viewer = _module
cityscapesViewer = _module
demo_cityscapes = _module
demo_mvd = _module
evaluation_cityscapes = _module
gan_augmention = _module
mapillary_vistas_loader = _module
models = _module
inceptionresnetv2 = _module
mixscaledensenet = _module
mobilenetv2aspp = _module
mobilenetv2exfuse = _module
mobilenetv2plus = _module
mobilenetv2share = _module
mobilenetv2vortex = _module
rfmobilenetv2context = _module
rfmobilenetv2plus = _module
rfshufflenetv2plus = _module
sedpshufflenet = _module
sewrnetv1 = _module
sewrnetv2 = _module
shufflenetv2plus = _module
modules = _module
_ext = _module
bn = _module
build = _module
context_encode = _module
dense = _module
dualpath = _module
exfuse = _module
functions = _module
group_norm = _module
misc = _module
residual = _module
rfblock = _module
net_viz = _module
guided_backprop = _module
layer_viz = _module
misc = _module
net_viz_pytorch = _module
visualize = _module
scripts = _module
cyclical_lr = _module
deploy_model = _module
loss = _module
metrics = _module
model_measure = _module
test_inplace = _module
train_auxiliary = _module
train_context = _module
train_inplace = _module
train_lovasz = _module
train_mixscale = _module
train_mobile = _module
train_mobile_mvd = _module
train_sedpnet = _module
train_share = _module
train_shuffle = _module
train_vortex = _module
utils = _module
yellowfin = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import scipy.misc as misc


from torch.utils import data


import time


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


from functools import partial


import torch.nn as nn


from collections import OrderedDict


import math


from torch.nn import init


from collections import Iterable


from itertools import repeat


import torch.autograd as autograd


import torch.cuda.comm as comm


from torch.autograd.function import once_differentiable


import tensorflow as tf


from torch.nn import ReLU


from torch.optim import SGD


from torchvision import models


import copy


from torch.optim import Optimizer


from functools import reduce


import random


from itertools import filterfalse


import logging


class BasicConv2d(nn.Module):
    """
        Define the basic conv-bn-relu block
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilate=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilate, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    """
        Define the 35x35 grid modules of Inception V4 network
        to replace later-half stem modules of InceptionResNet V2
    """

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    """
        The 35x35 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
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
    """
        The 35x35 to 17x17 reduction module
    """

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=1, padding=2, dilate=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=2, dilate=2), BasicConv2d(256, 384, kernel_size=3, stride=1, padding=2, dilate=2))
        self.branch2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """
        The 17x17 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 6), dilate=(1, 2)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(6, 0), dilate=(2, 1)))
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
    """
        The 17x17 to 8x8 reduction module
    """

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=1, padding=4, dilate=4))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=4, dilate=4))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=4, dilate=4), BasicConv2d(288, 320, kernel_size=3, stride=1, padding=4, dilate=4))
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """
        The 8x8 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 4), dilate=(1, 4)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(4, 0), dilate=(4, 1)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : nn.Module
            Module used as an activation function.
        kwargs
            All other arguments are forwarded to the `BatchNorm2d` constructor.
        """
        super(ABN, self).__init__(OrderedDict([('bn', nn.BatchNorm2d(num_features, **kwargs)), ('act', activation)]))


class ASPPInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(ASPPInPlaceABNBlock, self).__init__()
        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode='bilinear'))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv2_1', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[0], bias=False, groups=1, dilation=aspp_sec[0]))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv2_2', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[1], bias=False, groups=1, dilation=aspp_sec[1]))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv2_3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[2], bias=False, groups=1, dilation=aspp_sec[2]))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('norm_act', norm_act(5 * out_chs)), ('conv_down', nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups=1, dilation=1)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1(x), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        out = self.aspp_catdown(x)
        return out, self.upsampling(out)


class ModifiedSCSEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)), nn.ReLU(inplace=True), nn.Linear(int(channel // reduction), channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)


class InceptionResNetV2(nn.Module):

    def __init__(self, num_clases=19, in_size=(448, 896), aspp_out=512, fusion_out=64, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(InceptionResNetV2, self).__init__()
        self.num_clases = num_clases
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2, padding=1)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        if num_clases != 0:
            self.stg3_fusion = nn.Conv2d(192, fusion_out, kernel_size=1, stride=1, padding=0, bias=False)
            self.aspp = nn.Sequential(OrderedDict([('aspp', ASPPInPlaceABNBlock(1536, aspp_out, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, aspp_sec=aspp_sec))]))
            self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=aspp_out + fusion_out, reduction=16))
            self.score = nn.Sequential(OrderedDict([('conv', nn.Conv2d(aspp_out + fusion_out, num_clases, kernel_size=3, stride=1, padding=1, bias=True)), ('up', nn.Upsample(size=in_size, mode='bilinear'))]))

    def forward(self, x):
        c1 = self.conv2d_1a(x)
        c1 = self.conv2d_2a(c1)
        c1 = self.conv2d_2b(c1)
        c1 = self.maxpool_3a(c1)
        c1 = self.conv2d_3b(c1)
        c1 = self.conv2d_4a(c1)
        c2 = self.maxpool_5a(c1)
        c2 = self.mixed_5b(c2)
        c2 = self.repeat(c2)
        c2 = self.mixed_6a(c2)
        c2 = self.repeat_1(c2)
        c2 = self.mixed_7a(c2)
        c2 = self.repeat_2(c2)
        c2 = self.block8(c2)
        c2 = self.conv2d_7b(c2)
        if self.num_clases != 0:
            c2 = self.score_se(torch.cat([self.aspp(c2)[1], self.stg3_fusion(c1)], dim=1))
            return self.score(c2)
        else:
            return c2


class DenseModule(nn.Module):

    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers, dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DenseModule, self).__init__()
        self.n_layers = n_layers
        self.mid_out = int(in_chns * squeeze_ratio)
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns
            """
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, self.mid_out, 1, bias=False))
            ])))
            """
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(self.last_channel)), ('conv', nn.Conv2d(self.last_channel, out_chns, kernel_size=3, stride=1, padding=dilate, dilation=dilate, bias=False))])))

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


ACT_LEAKY_RELU = 'leaky_relu'


ACT_ELU = 'elu'


ACT_NONE = 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _ext.leaky_relu_backward_cuda(x, dx, ctx.slope)
        _ext.leaky_relu_cuda(x, 1.0 / ctx.slope)
    elif ctx.activation == ACT_ELU:
        _ext.elu_backward_cuda(x, dx)
        _ext.elu_inv_cuda(x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _ext.leaky_relu_cuda(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _ext.elu_cuda(x)
    elif ctx.activation == ACT_NONE:
        pass


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABN(autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        n = _count_samples(x)
        if ctx.training:
            mean = x.new().resize_as_(running_mean)
            var = x.new().resize_as_(running_var)
            _check_contiguous(x, mean, var)
            _ext.bn_mean_var_cuda(x, mean, var)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * n / (n - 1))
        else:
            mean, var = running_mean, running_var
        _check_contiguous(x, mean, var, weight, bias)
        _ext.bn_forward_cuda(x, mean, var, weight if weight is not None else x.new(), bias if bias is not None else x.new(), x, x, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, weight, bias, running_mean, running_var)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, weight, bias, running_mean, running_var = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz)
        else:
            dx = None
        if ctx.needs_input_grad[1]:
            dweight = dz.new().resize_as_(running_mean).zero_()
        else:
            dweight = None
        if ctx.needs_input_grad[2]:
            dbias = dz.new().resize_as_(running_mean).zero_()
        else:
            dbias = None
        if ctx.training:
            edz = dz.new().resize_as_(running_mean)
            eydz = dz.new().resize_as_(running_mean)
            _check_contiguous(z, dz, weight, bias, edz, eydz)
            _ext.bn_edz_eydz_cuda(z, dz, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz)
        else:
            edz = dz.new().resize_as_(running_mean).zero_()
            eydz = dz.new().resize_as_(running_mean).zero_()
        _check_contiguous(dz, z, ctx.var, weight, bias, edz, eydz, dx, dweight, dbias)
        _ext.bn_backard_cuda(dz, z, ctx.var, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, dx if dx is not None else dz.new(), dweight if dweight is not None else dz.new(), dbias if dbias is not None else dz.new(), ctx.eps)
        del ctx.var
        return dx, dweight, dbias, None, None, None, None, None, None, None


class SCSEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)), nn.ReLU(inplace=True), nn.Linear(int(channel // reduction), channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)
        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class MixedScaleDenseNet(nn.Module):
    """
    Mixed Scale Dense Network
    """

    def __init__(self, n_class=19, in_size=(448, 896), num_layers=128, in_chns=32, squeeze_ratio=1.0 / 32, out_chns=1, dilate_sec=(1, 2, 4, 8, 4, 2), aspp_sec=(24, 48, 72), norm_act=InPlaceABN):
        """
        MixedScaleDenseNet: Mixed Scale Dense Network

        :param n_class:    (int) Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param num_layers: (int) Number of layers used in the mixed scale dense block/stage
        :param in_chns:    (int) Input channels of the mixed scale dense block/stage
        :param out_chns:   (int) Output channels of each Conv used in the mixed scale dense block/stage
        :param dilate_sec: (tuple) Dilation rates used in the mixed scale dense block/stage
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        :param norm_act:   (object) Batch Norm Activation Type
        """
        super(MixedScaleDenseNet, self).__init__()
        self.n_classes = n_class
        self.conv_in = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels=3, out_channels=in_chns, kernel_size=7, stride=2, padding=3, bias=False)), ('norm', norm_act(in_chns)), ('pool', nn.MaxPool2d(3, stride=2, padding=1))]))
        self.dense = DenseModule(in_chns, squeeze_ratio, out_chns, num_layers, dilate_sec=dilate_sec, norm_act=norm_act)
        self.last_channel = self.dense.out_channels
        self.feat_out = norm_act(self.last_channel)
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_classes != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, self.last_channel, feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)), aspp_sec=aspp_sec, norm_act=norm_act))
            self.score_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(self.last_channel)), ('conv.1', nn.Conv2d(self.last_channel, self.last_channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(self.last_channel)), ('conv.2', nn.Conv2d(self.last_channel, self.n_classes, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))

    def forward(self, x):
        x = self.out_se(self.feat_out(self.dense(self.conv_in(x))))
        if self.n_classes != 0:
            return self.score(self.score_se(self.aspp(x)[1]))
        else:
            return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True))

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)


class SDASPPInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(SDASPPInPlaceABNBlock, self).__init__()
        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode='bilinear'))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('dconv2_1', nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[0], bias=False, groups=in_chs, dilation=aspp_sec[0])), ('pconv2_1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('dconv2_2', nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[1], bias=False, groups=in_chs, dilation=aspp_sec[1])), ('pconv2_2', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('dconv2_3', nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[2], bias=False, groups=in_chs, dilation=aspp_sec[2])), ('pconv2_3', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('norm_act', norm_act(5 * out_chs)), ('conv_down', nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups=1, dilation=1)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1(x), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        return self.upsampling(self.aspp_catdown(x))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True), nn.ReLU(inplace=True))


class MobileNetV2ASPP(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2ASPP, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(ModifiedSCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.sdaspp = nn.Sequential(SDASPPInPlaceABNBlock(self.last_channel, out_sec, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), aspp_sec=aspp_sec, norm_act=norm_act))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
            """
            self.score = nn.Sequential(OrderedDict([("norm", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("dconv", nn.Conv2d(out_sec + in_stag2_up_chs,
                                                                        out_sec + in_stag2_up_chs,
                                                                        kernel_size=3, stride=1, padding=2,
                                                                        groups=out_sec + in_stag2_up_chs,
                                                                        dilation=2, bias=False)),
                                                    ("pconv", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
                                                                        kernel_size=1, stride=1, padding=0,
                                                                        bias=True)),
                                                    ("up", nn.Upsample(size=in_size, mode='bilinear'))]))
            """
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, padding=1)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.n_class != 0:
            de_stg1 = self.sdaspp(stg8)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg8


class RFBlock(nn.Module):

    def __init__(self, in_chs, out_chs, scale=0.1, feat_res=(56, 112), aspp_sec=(12, 24, 36), up_ratio=2, norm_act=InPlaceABN):
        super(RFBlock, self).__init__()
        self.scale = scale
        self.down_chs = nn.Sequential(OrderedDict([('norm_act', norm_act(in_chs)), ('down_conv1x1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False))]))
        self.gave_pool = nn.Sequential(OrderedDict([('norm_act', norm_act(out_chs)), ('gavg', nn.AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode='bilinear'))]))
        self.branch0 = nn.Sequential(OrderedDict([('norm_act', norm_act(out_chs)), ('conv1x1', nn.Conv2d(out_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)), ('norm_act', norm_act(out_chs)), ('aconv1', nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1, dilation=1, bias=False))]))
        self.branch1 = nn.Sequential(OrderedDict([('norm_act', norm_act(out_chs)), ('conv1x3', nn.Conv2d(out_chs, out_chs // 2 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)), ('norm_act', norm_act(out_chs // 2 * 3)), ('conv3x1', nn.Conv2d(out_chs // 2 * 3, out_chs, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)), ('norm_act', norm_act(out_chs)), ('aconv3', nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[0], dilation=aspp_sec[0], bias=False))]))
        self.branch2 = nn.Sequential(OrderedDict([('norm_act', norm_act(out_chs)), ('conv1x5', nn.Conv2d(out_chs, out_chs // 2 * 3, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False)), ('norm_act', norm_act(out_chs // 2 * 3)), ('conv5x1', nn.Conv2d(out_chs // 2 * 3, out_chs, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False)), ('norm_act', norm_act(out_chs)), ('aconv5', nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[1], dilation=aspp_sec[1], bias=False))]))
        self.branch3 = nn.Sequential(OrderedDict([('norm_act', norm_act(out_chs)), ('conv1x7', nn.Conv2d(out_chs, out_chs // 2 * 3, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False)), ('norm_act', norm_act(out_chs // 2 * 3)), ('conv7x1', nn.Conv2d(out_chs // 2 * 3, out_chs, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False)), ('norm_act', norm_act(out_chs)), ('aconv7', nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[2], dilation=aspp_sec[2], bias=False))]))
        self.conv_linear = nn.Sequential(OrderedDict([('conv1x1_linear', nn.Conv2d(out_chs * 5, out_chs, kernel_size=1, stride=1, padding=0, bias=False))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        down = self.down_chs(x)
        out = torch.cat([self.gave_pool(down.clone()), self.branch0(down.clone()), self.branch1(down.clone()), self.branch2(down.clone()), self.branch3(down.clone())], dim=1)
        return self.upsampling(torch.add(self.conv_linear(out), self.scale, down))


class SemanticSupervision(nn.Module):

    def __init__(self, in_chns, out_chns):
        super(SemanticSupervision, self).__init__()
        self.out_chns = out_chns
        self.semantic = nn.Sequential(OrderedDict([('conv1x7', nn.Conv2d(in_chns, in_chns // 2 * 3, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False)), ('norm1', nn.BatchNorm2d(in_chns // 2 * 3)), ('act1', nn.LeakyReLU(negative_slope=0.1, inplace=True)), ('conv7x1', nn.Conv2d(in_chns // 2 * 3, out_chns // 2 * 3, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False)), ('norm2', nn.BatchNorm2d(out_chns // 2 * 3)), ('act2', nn.LeakyReLU(negative_slope=0.1, inplace=True))]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_chns // 2 * 3, out_chns)

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        se = self.semantic(x)
        se = self.avg_pool(se).view(bahs, self.out_chns // 2 * 3)
        se = self.classifier(se)
        return se


class MobileNetV2ExFuse(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, norm_act=InPlaceABN, traval='train'):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2ExFuse, self).__init__()
        self.n_class = n_class
        self.traval = traval
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.rfblock = nn.Sequential(RFBlock(in_chs=self.last_channel, out_chs=out_sec, scale=1.0, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, norm_act=norm_act))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
            self.sesuper1 = SemanticSupervision(in_chns=self.interverted_residual_setting[0][1], out_chns=self.n_class)
            self.sesuper2 = SemanticSupervision(in_chns=self.interverted_residual_setting[1][1], out_chns=self.n_class)
            self.sesuper3 = SemanticSupervision(in_chns=self.interverted_residual_setting[2][1], out_chns=self.n_class)
            self.sesuper4 = SemanticSupervision(in_chns=self.interverted_residual_setting[3][1], out_chns=self.n_class)
            self.sesuper5 = SemanticSupervision(in_chns=self.interverted_residual_setting[4][1], out_chns=self.n_class)
            self.sesuper6 = SemanticSupervision(in_chns=self.interverted_residual_setting[5][1], out_chns=self.n_class)
            self.sesuper7 = SemanticSupervision(in_chns=self.interverted_residual_setting[6][1], out_chns=self.n_class)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.traval == 'train' and self.n_class != 0:
            de_stg1 = self.rfblock(stg8)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            enc1 = self.sesuper1(stg1)
            enc2 = self.sesuper2(stg2)
            enc3 = self.sesuper3(stg3)
            enc4 = self.sesuper4(stg4)
            enc5 = self.sesuper5(stg5)
            enc6 = self.sesuper6(stg6)
            enc7 = self.sesuper7(stg7)
            return enc1, enc2, enc3, enc4, enc5, enc6, enc7, net_out
        elif self.traval == 'train' and self.n_class != 0:
            de_stg1 = self.rfblock(stg8)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            return self.score(de_stg1)
        else:
            return stg8


class MobileNetV2Plus(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2Plus, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, out_sec, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), aspp_sec=aspp_sec, norm_act=norm_act))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.n_class != 0:
            de_stg1 = self.aspp(stg8)[1]
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg8


class MobileNetV2Share(nn.Module):

    def __init__(self, n_class=19, in_size=(896, 17), width_mult=1.0, out_sec=256, aspp_sec=(24, 48, 72), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2Share, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, out_sec, feat_res=(int(in_size[0] / 16), int(in_size[1] / 16)), aspp_sec=aspp_sec, norm_act=norm_act))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    @staticmethod
    def __split_cat(x):
        h_up, h_down = x.chunk(chunks=2, dim=2)
        w_left_up, w_right_up = h_up.chunk(chunks=2, dim=3)
        w_left_down, w_right_down = h_down.chunk(chunks=2, dim=3)
        return torch.cat([w_left_up, w_right_up, w_left_down, w_right_down], dim=0)

    @staticmethod
    def __recat_feat(x, bs=2):
        batch_size, num_channels, height, width = x.data.size()
        batches = x.chunk(chunks=batch_size // bs, dim=0)
        out = torch.cat([torch.cat([batches[0], batches[1]], dim=3), torch.cat([batches[2], batches[3]], dim=3)], dim=2)
        return out

    def forward(self, x):
        batch_size = x.size(0)
        x = self.__split_cat(x)
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.n_class != 0:
            de_stg1 = self.aspp(stg8)[1]
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            de_stg1 = self.__recat_feat(de_stg1, bs=batch_size)
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg8


class VortexPooling(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, rate=(3, 9, 27)):
        super(VortexPooling, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.AdaptiveAvgPool2d((1, 1))), ('conv1x1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode='bilinear')), ('bn0', nn.BatchNorm2d(num_features=out_chs))]))
        self.conv3x3 = nn.Sequential(OrderedDict([('conv3x3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False, groups=1, dilation=1)), ('bn3x3', nn.BatchNorm2d(num_features=out_chs))]))
        self.vortex_bra1 = nn.Sequential(OrderedDict([('avg_pool', nn.AvgPool2d(kernel_size=rate[0], stride=1, padding=int((rate[0] - 1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=rate[0], bias=False, groups=1, dilation=rate[0])), ('bn3x3', nn.BatchNorm2d(num_features=out_chs))]))
        self.vortex_bra2 = nn.Sequential(OrderedDict([('avg_pool', nn.AvgPool2d(kernel_size=rate[1], stride=1, padding=int((rate[1] - 1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=rate[1], bias=False, groups=1, dilation=rate[1])), ('bn3x3', nn.BatchNorm2d(num_features=out_chs))]))
        self.vortex_bra3 = nn.Sequential(OrderedDict([('avg_pool', nn.AvgPool2d(kernel_size=rate[2], stride=1, padding=int((rate[2] - 1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=rate[2], bias=False, groups=1, dilation=rate[2])), ('bn3x3', nn.BatchNorm2d(num_features=out_chs))]))
        self.vortex_catdown = nn.Sequential(OrderedDict([('conv_down', nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups=1, dilation=1)), ('bn_down', nn.BatchNorm2d(num_features=out_chs)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        out = torch.cat([self.gave_pool(x), self.conv3x3(x), self.vortex_bra1(x), self.vortex_bra2(x), self.vortex_bra3(x)], dim=1)
        out = self.vortex_catdown(out)
        return self.upsampling(out)


class MobileNetV2Vortex(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, rate_sec=(3, 9, 27), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2Vortex, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(ModifiedSCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.vortex = nn.Sequential(VortexPooling(self.last_channel, out_sec, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), rate=rate_sec))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
            """
            self.score = nn.Sequential(OrderedDict([("norm", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("dconv", nn.Conv2d(out_sec + in_stag2_up_chs,
                                                                        out_sec + in_stag2_up_chs,
                                                                        kernel_size=3, stride=1, padding=2,
                                                                        groups=out_sec + in_stag2_up_chs,
                                                                        dilation=2, bias=False)),
                                                    ("pconv", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
                                                                        kernel_size=1, stride=1, padding=0,
                                                                        bias=True)),
                                                    ("up", nn.Upsample(size=in_size, mode='bilinear'))]))
            """
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, padding=1)
        stg7 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.n_class != 0:
            de_stg1 = self.vortex(stg7)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg7


class ContextEncodeDropInplaceABN(nn.Module):

    def __init__(self, channel, K=16, reduction=4, norm_act=InPlaceABN):
        super(ContextEncodeDropInplaceABN, self).__init__()
        out_channel = int(channel / reduction)
        self.pre_abn = norm_act(channel)
        self.context_enc = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0), norm_act(out_channel), encoding.nn.EncodingDrop(D=out_channel, K=K), encoding.nn.View(-1, out_channel * K), encoding.nn.Normalize())
        self.channel_se = nn.Sequential(nn.Linear(out_channel * K, channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        pre_x = self.pre_abn(x.clone())
        encode = self.context_enc(pre_x)
        chn_se = self.channel_se(encode).view(batch_size, num_channels, 1, 1)
        spa_se = self.spatial_se(pre_x)
        return encode, torch.mul(torch.mul(x, spa_se), chn_se)


class MobileNetV2Context(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, context=(32, 4), aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param context:   (tuple) K and reduction
        """
        super(MobileNetV2Context, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.context1 = ContextEncodeDropInplaceABN(channel=self.last_channel, K=context[0], reduction=context[1], norm_act=norm_act)
        self.se_loss1 = nn.Sequential(OrderedDict([('linear', nn.Linear(int(self.last_channel / context[1]) * context[0], self.n_class))]))
        self.rfblock = nn.Sequential(RFBlock(in_chs=self.last_channel, out_chs=out_sec, scale=1.0, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, aspp_sec=aspp_sec, norm_act=norm_act))
        if self.n_class != 0:
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.context2 = ContextEncodeDropInplaceABN(channel=out_sec + in_stag2_up_chs, K=context[0], reduction=context[1], norm_act=norm_act)
            self.se_loss2 = nn.Sequential(OrderedDict([('linear', nn.Linear(int((out_sec + in_stag2_up_chs) / context[1]) * context[0], self.n_class))]))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec)), ('conv.2', nn.Conv2d(out_sec, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
            """

            self.score = nn.Sequential(OrderedDict([("norm", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("conv", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
                                                                       kernel_size=1, stride=1, padding=0,
                                                                       bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))
            """
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)
        enc1, stg8 = self.context1(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        stg8 = self.rfblock(stg8)
        if self.n_class != 0:
            enc2, stg8 = self.context2(torch.cat([stg8, stg2, stg1_1], dim=1))
            return self.se_loss1(enc1), self.se_loss2(enc2), self.score(stg8)
        else:
            return stg8


class RFMobileNetV2Plus(nn.Module):

    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.0, out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(RFMobileNetV2Plus, self).__init__()
        self.n_class = n_class
        self.interverted_residual_setting = [[1, 16, 1, 1, 1], [6, 24, 2, 2, 1], [6, 32, 3, 2, 1], [6, 64, 4, 1, 2], [6, 96, 3, 1, 4], [6, 160, 3, 1, 8], [6, 320, 1, 1, 16]]
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.input_size = in_size
        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([('conv1', conv_bn(inp=3, oup=input_channel, stride=2))]))
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t)))
                else:
                    blocks.append(('block%d' % (block_id + 1), InvertedResidual(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t)))
                input_channel = output_channel
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1
        org_last_chns = self.interverted_residual_setting[0][1] + self.interverted_residual_setting[1][1] + self.interverted_residual_setting[2][1] + self.interverted_residual_setting[3][1] + self.interverted_residual_setting[4][1] + self.interverted_residual_setting[5][1] + self.interverted_residual_setting[6][1]
        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.rfblock = nn.Sequential(RFBlock(in_chs=self.last_channel, out_chs=out_sec, scale=1.0, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, norm_act=norm_act))
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg1 = self.mod1(x)
        stg1 = self.mod2(stg1)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)
        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)
        stg7 = self.mod8(stg6)
        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)
        stg8 = self.out_se(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        if self.n_class != 0:
            de_stg1 = self.rfblock(stg8)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg2, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg8


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def conv3x3(in_channels, out_channels, stride=1, bias=True, groups=1, dilate=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilate, bias=bias, groups=groups, dilation=dilate)


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, dilate=1, grouped_conv=True, combine='add', up=False):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
            self.dilate = dilate
            self.up = False
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 1 if up is True else 2
            self.dilate = dilate if up is True else 1
            self.up = up
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError('Cannot combine tensors with "{}"Only "add" and "concat" aresupported'.format(self.combine))
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.in_channels, self.bottleneck_channels, self.first_1x1_groups, batch_norm=True, relu=True)
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.bottleneck_channels, stride=self.depthwise_stride, groups=self.bottleneck_channels, dilate=self.dilate)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels, self.out_channels, self.groups, batch_norm=True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), dim=1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
            if self.up is True:
                residual = F.upsample(residual, scale_factor=2, mode='bilinear')
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class RFShuffleNetV2Plus(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, n_class=19, groups=3, in_channels=3, in_size=(448, 896), out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 19 for ImageNet.

        """
        super(RFShuffleNetV2Plus, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.n_class = n_class
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError("""{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2, dilate=2)
        self.stage3 = self._make_stage(3, dilate=4)
        self.stage4 = self._make_stage(4, dilate=8)
        self.last_channel = 2 * self.stage_out_channels[1] + self.stage_out_channels[2] + self.stage_out_channels[3] + self.stage_out_channels[4]
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.rfblock = nn.Sequential(RFBlock(in_chs=self.last_channel, out_chs=out_sec, scale=1.0, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, aspp_sec=aspp_sec, norm_act=norm_act))
            in_stag2_up_chs = 2 * self.stage_out_channels[1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_stage(self, stage, dilate=1):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)
        grouped_conv = stage > 2
        if stage >= 3:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self.stage_out_channels[stage], groups=self.groups, dilate=dilate, grouped_conv=grouped_conv, combine='concat', up=True)
        else:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self.stage_out_channels[stage], groups=self.groups, dilate=1, grouped_conv=grouped_conv, combine='concat', up=False)
        modules[stage_name + '_0'] = first_module
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + '_{}'.format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.stage_out_channels[stage], groups=self.groups, dilate=dilate, grouped_conv=True, combine='add', up=False)
            modules[name] = module
        return nn.Sequential(modules)

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg0 = self.conv1(x)
        stg1 = self.maxpool(stg0)
        stg2 = self.stage2(stg1)
        stg3 = self.stage3(stg2)
        stg4 = self.stage4(stg3)
        stg1_1 = F.avg_pool2d(input=stg0, kernel_size=3, stride=2, padding=1)
        stg1_2 = F.avg_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)
        stg1_3 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)
        stg5 = self.out_se(torch.cat([stg2, stg3, stg4, stg1_2, stg1_3], dim=1))
        if self.n_class != 0:
            de_stg1 = self.rfblock(stg5)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg1, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg5


class CatInPlaceABN(nn.Module):
    """
    Block for concat the two output tensor of feature net
    """

    def __init__(self, in_chs, norm_act=ABN):
        super(CatInPlaceABN, self).__init__()
        self.norm_act = norm_act(in_chs)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x = self.norm_act(x)
        return x


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(nn.Linear(channel, int(channel / reduction)), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Linear(int(channel / reduction), channel), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        y = self.avg_pool(x).view(bahs, chs)
        y = self.fcs(y).view(bahs, chs, 1, 1)
        return torch.mul(x, y)


class DualPathInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups=1, dilation=1, block_type='normal', norm_act=ABN):
        super(DualPathInPlaceABNBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.dilation = dilation
        self.groups = groups
        self.inc = inc
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = nn.Sequential(OrderedDict([('conv1x1_w_s2_bn', norm_act(in_chs)), ('conv1x1_w_s2', nn.Conv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1, stride=2, padding=0, groups=self.groups, dilation=1, bias=False))]))
            else:
                self.c1x1_w_s1 = nn.Sequential(OrderedDict([('conv1x1_w_s1_bn', norm_act(in_chs)), ('conv1x1_w_s1', nn.Conv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1, stride=1, padding=0, groups=self.groups, dilation=1, bias=False))]))
        self.c1x1_a = nn.Sequential(OrderedDict([('conv1x1_a_bn', norm_act(in_chs)), ('conv1x1_a', nn.Conv2d(in_chs, num_1x1_a, kernel_size=1, stride=1, padding=0, groups=self.groups, dilation=1, bias=False))]))
        if self.dilation > 1 and self.key_stride == 1:
            self.c3x3_b = nn.Sequential(OrderedDict([('conv3x3_b_bn', norm_act(num_1x1_a)), ('conv3x3_b', nn.Conv2d(num_1x1_a, num_3x3_b, kernel_size=3, stride=1, padding=dilation, groups=num_1x1_a, dilation=self.dilation, bias=False))]))
        else:
            self.c3x3_b = nn.Sequential(OrderedDict([('conv3x3_b_bn', norm_act(num_1x1_a)), ('conv3x3_b', nn.Conv2d(num_1x1_a, num_3x3_b, kernel_size=3, stride=self.key_stride, padding=1, groups=num_1x1_a, dilation=1, bias=False))]))
        self.c1x1_c = nn.Sequential(OrderedDict([('conv1x1_c_bn', norm_act(num_3x3_b)), ('conv1x1_c', nn.Conv2d(num_3x3_b, num_1x1_c + inc, kernel_size=1, stride=1, padding=0, groups=self.groups, dilation=1, bias=False)), ('se_block', SEBlock(num_1x1_c + inc, 16)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in.clone())
            else:
                x_s = self.c1x1_w_s1(x_in.clone())
            x_s = self._channel_shuffle(x_s, self.groups)
            x_s = torch.split(x_s, self.num_1x1_c, dim=1)
        else:
            x_s = x
        x_r = self.c1x1_a(x_in)
        x_r = self._channel_shuffle(x_r, self.groups)
        x_r = self.c3x3_b(x_r)
        x_r = self.c1x1_c(x_r)
        x_r = self._channel_shuffle(x_r, self.groups)
        x_r = torch.split(x_r, self.num_1x1_c, dim=1)
        resid = torch.add(x_s[0], 1, x_r[0])
        dense = torch.cat([x_s[1], x_r[1]], dim=1)
        return resid, dense


class SEDPNShuffleNet(nn.Module):

    def __init__(self, small=False, classes=19, in_size=(448, 896), num_init_features=64, k_r=96, groups=4, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), out_sec=(512, 256, 128), dil_sec=(1, 1, 1, 2, 4), aspp_sec=(7, 14, 21), norm_act=ABN):
        super(SEDPNShuffleNet, self).__init__()
        bw_factor = 1 if small else 4
        if small:
            self.encode_in = nn.Sequential(OrderedDict([('conv_in', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)), ('bn_in', norm_act(num_init_features)), ('pool_in', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        else:
            self.encode_in = nn.Sequential(OrderedDict([('conv_in', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('bn_in', norm_act(num_init_features)), ('pool_in', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        conv1x1c_ch = 64 * bw_factor
        inc = inc_sec[0]
        conv1x1a_ch = k_r * conv1x1c_ch // (64 * bw_factor)
        conv3x3b_ch = conv1x1a_ch
        encode_blocks1 = OrderedDict()
        encode_blocks1['conv2_1'] = DualPathInPlaceABNBlock(num_init_features, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[0], 'proj', norm_act=norm_act)
        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, k_sec[0] + 1):
            encode_blocks1['conv2_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[0], 'normal', norm_act=norm_act)
            in_chs += inc
        self.encode_stg1 = nn.Sequential(encode_blocks1)
        conv1x1c_ch = 128 * bw_factor
        inc = inc_sec[1]
        conv1x1a_ch = k_r * conv1x1c_ch // (64 * bw_factor)
        conv3x3b_ch = conv1x1a_ch
        encode_blocks2 = OrderedDict()
        encode_blocks2['conv3_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[1], 'down', norm_act=norm_act)
        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, k_sec[1] + 1):
            encode_blocks2['conv3_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[1], 'normal', norm_act=norm_act)
            in_chs += inc
        self.encode_stg2 = nn.Sequential(encode_blocks2)
        conv1x1c_ch = 256 * bw_factor
        inc = inc_sec[2]
        conv1x1a_ch = k_r * conv1x1c_ch // (64 * bw_factor)
        conv3x3b_ch = conv1x1a_ch
        encode_blocks3 = OrderedDict()
        encode_blocks3['conv4_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[2], 'down', norm_act=norm_act)
        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, int(k_sec[2] / 2) + 1):
            encode_blocks3['conv4_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[2], 'normal', norm_act=norm_act)
            in_chs += inc
        for i in range(int(k_sec[2] / 2) + 1, k_sec[2] + 1):
            encode_blocks3['conv4_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[3], 'normal', norm_act=norm_act)
            in_chs += inc
        self.encode_stg3 = nn.Sequential(encode_blocks3)
        conv1x1c_ch = 512 * bw_factor
        inc = inc_sec[3]
        conv1x1a_ch = k_r * conv1x1c_ch // (64 * bw_factor)
        conv3x3b_ch = conv1x1a_ch
        encode_blocks4 = OrderedDict()
        encode_blocks4['conv5_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[4], 'proj', norm_act=norm_act)
        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, k_sec[3] + 1):
            encode_blocks4['conv5_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch, conv1x1c_ch, inc, groups, dil_sec[4], 'normal', norm_act=norm_act)
            in_chs += inc
        encode_blocks4['conv5_bn_ac'] = CatInPlaceABN(in_chs)
        self.encode_stg4 = nn.Sequential(encode_blocks4)
        self.aspp1 = nn.Sequential(OrderedDict([('aspp1', ASPPInPlaceABNBlock(in_chs, out_sec[0], feat_res=(int(in_size[0] / 16), int(in_size[1] / 16)), aspp_sec=aspp_sec, norm_act=norm_act))]))
        self.aspp2_in = nn.Sequential(OrderedDict([('aspp2_in', CatInPlaceABN(704, norm_act=norm_act))]))
        self.aspp2 = nn.Sequential(OrderedDict([('aspp2', ASPPInPlaceABNBlock(out_sec[0] + 704, out_sec[1], feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), aspp_sec=(aspp_sec[0] * 2, aspp_sec[1] * 2, aspp_sec[2] * 2), norm_act=norm_act))]))
        self.aspp3_in = nn.Sequential(OrderedDict([('aspp3_in', CatInPlaceABN(336, norm_act=norm_act))]))
        self.aspp3 = nn.Sequential(OrderedDict([('aspp3', ASPPInPlaceABNBlock(out_sec[1] + 336, out_sec[2], feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)), up_ratio=4, aspp_sec=(aspp_sec[0] * 4, aspp_sec[1] * 4, aspp_sec[2] * 4), norm_act=norm_act))]))
        self.score1 = nn.Sequential(OrderedDict([('score1', nn.Conv2d(out_sec[0], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se1_classes', SEBlock(classes, 4)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self.score2 = nn.Sequential(OrderedDict([('score2', nn.Conv2d(out_sec[1], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se2_classes', SEBlock(classes, 4)), ('up2', nn.Upsample(size=in_size, mode='bilinear'))]))
        self.score3 = nn.Sequential(OrderedDict([('score3', nn.Conv2d(out_sec[2], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se3_classes', SEBlock(classes, 4))]))
        self.score4 = nn.Sequential(OrderedDict([('score4_norm', norm_act(classes)), ('score4', nn.Conv2d(classes, classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se4_classes', SEBlock(classes, 4))]))

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        en_in = self.encode_in(x)
        en_stg1 = self.encode_stg1(en_in)
        en_stg2 = self.encode_stg2(en_stg1)
        en_stg3 = self.encode_stg3(en_stg2)
        en_stg4 = self.encode_stg4(en_stg3)
        out_stg1, de_stg1 = self.aspp1(en_stg4)
        out_stg2, de_stg2 = self.aspp2(self._channel_shuffle(torch.cat([de_stg1, self.aspp2_in(en_stg2)], dim=1), 2))
        de_stg3 = self.aspp3(self._channel_shuffle(torch.cat([de_stg2, self.aspp3_in(en_stg1)], dim=1), 2))[1]
        out_stg1 = self.score1(out_stg1)
        out_stg2 = self.score2(out_stg2)
        out_stg3 = self.score3(de_stg3)
        out_stg4 = self.score4(torch.max(torch.max(out_stg1, out_stg2), out_stg3))
        return out_stg1, out_stg2, out_stg3, out_stg4


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, is_se=False, dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            if not is_se:
                layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
            else:
                layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)), ('se_block', SEBlock(channels[2], 16))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class SEWiderResNetV1(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0, dilation=False, is_se=True, in_size=(448, 896), out_sec=(512, 256, 128), aspp_sec=(12, 24, 36)):
        """
        Wider ResNet with pre-activation (identity mapping) and Squeeze & Excitation(SE) blocks

        :param structure: (list of int) Number of residual blocks in each of the six modules of the network.
        :param norm_act:  (callable) Function to create normalization / activation Module.
        :param classes:   (int) Not `0` for segmentation task
        :param dilation:  (bool) `True` for segmentation task
        :param is_se:     (bool) Use Squeeze & Excitation (SE) or not
        :param in_size:   (tuple of int) Size of the input image
        :param out_sec:   (tuple of int) Number of channels of the ASPP output
        :param aspp_sec:  (tuple of int) Dilation rate used in ASPP
        """
        super(SEWiderResNetV1, self).__init__()
        self.structure = structure
        self.dilation = dilation
        self.classes = classes
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id == 4:
                        dil = 4
                    elif mod_id == 5:
                        dil = 8
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.2)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.3)
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop, is_se=is_se)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.aspp1 = nn.Sequential(OrderedDict([('aspp1', ASPPInPlaceABNBlock(channels[5][2], out_sec[0], feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), aspp_sec=aspp_sec))]))
            self.aspp2 = nn.Sequential(OrderedDict([('aspp2', ASPPInPlaceABNBlock(out_sec[0] + 256, out_sec[1], feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)), aspp_sec=(aspp_sec[0] * 2, aspp_sec[1] * 2, aspp_sec[2] * 2)))]))
            self.aspp3 = nn.Sequential(OrderedDict([('aspp3', ASPPInPlaceABNBlock(out_sec[1] + 128, out_sec[2], feat_res=(int(in_size[0] / 2), int(in_size[1] / 2)), aspp_sec=(aspp_sec[0] * 4, aspp_sec[1] * 4, aspp_sec[2] * 4)))]))
            self.score1 = nn.Sequential(OrderedDict([('score1', nn.Conv2d(out_sec[0], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se1_classes', SEBlock(classes, 4)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
            self.score2 = nn.Sequential(OrderedDict([('score2', nn.Conv2d(out_sec[1], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se2_classes', SEBlock(classes, 4)), ('up2', nn.Upsample(size=in_size, mode='bilinear'))]))
            self.score3 = nn.Sequential(OrderedDict([('score3', nn.Conv2d(out_sec[2], classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se3_classes', SEBlock(classes, 4))]))
            self.score4 = nn.Sequential(OrderedDict([('score4_norm', norm_act(classes)), ('score4', nn.Conv2d(classes, classes, kernel_size=1, stride=1, padding=0, bias=True)), ('se4_classes', SEBlock(classes, 4))]))

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, img):
        stg1 = self.mod1(img)
        stg2 = self.mod2(self.pool2(stg1))
        stg3 = self.mod3(self.pool3(stg2))
        stg4 = self.mod4(stg3)
        stg4 = self.mod5(stg4)
        stg4 = self.mod6(stg4)
        stg4 = self.mod7(stg4)
        stg4 = self.bn_out(stg4)
        if self.classes != 0:
            out_stg1, de_stg1 = self.aspp1(stg4)
            out_stg2, de_stg2 = self.aspp2(self._channel_shuffle(torch.cat([de_stg1, stg3], dim=1), 3))
            de_stg3 = self.aspp3(self._channel_shuffle(torch.cat([de_stg2, stg2], dim=1), 3))[1]
            out_stg1 = self.score1(out_stg1)
            out_stg2 = self.score2(out_stg2)
            out_stg3 = self.score3(de_stg3)
            out_stg4 = self.score4(torch.max(torch.max(out_stg1, out_stg2), out_stg3))
            return out_stg1, out_stg2, out_stg3, out_stg4
        else:
            return stg4


class SEWiderResNetV2(nn.Module):

    def __init__(self, structure, norm_act=ABN, classes=0, dilation=True, is_se=True, in_size=(448, 896), aspp_out=512, fusion_out=64, aspp_sec=(12, 24, 36)):
        """
        Wider ResNet with pre-activation (identity mapping) and Squeeze & Excitation(SE) blocks

        :param structure: (list of int) Number of residual blocks in each of the six modules of the network.
        :param norm_act:  (callable) Function to create normalization / activation Module.
        :param classes:   (int) Not `0` for segmentation task
        :param dilation:  (bool) `True` for segmentation task
        :param is_se:     (bool) Use Squeeze & Excitation (SE) or not
        :param in_size:   (tuple of int) Size of the input image
        :param out_sec:   (tuple of int) Number of channels of the ASPP output
        :param aspp_sec:  (tuple of int) Dilation rate used in ASPP
        """
        super(SEWiderResNetV2, self).__init__()
        self.structure = structure
        self.dilation = dilation
        self.classes = classes
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id == 4:
                        dil = 4
                    elif mod_id == 5:
                        dil = 8
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.2)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.3)
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1), IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop, is_se=is_se)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.stg3_fusion = nn.Conv2d(channels[1][1], fusion_out, kernel_size=1, stride=1, padding=0, bias=False)
            self.aspp = nn.Sequential(OrderedDict([('aspp', ASPPInPlaceABNBlock(channels[5][2], aspp_out, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), up_ratio=2, aspp_sec=aspp_sec))]))
            self.score = nn.Sequential(OrderedDict([('conv', nn.Conv2d(aspp_out + fusion_out, classes, kernel_size=3, stride=1, padding=1, bias=True)), ('up', nn.Upsample(size=in_size, mode='bilinear'))]))

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, img):
        stg1 = self.mod1(img)
        stg2 = self.mod2(self.pool2(stg1))
        stg3 = self.mod3(self.pool3(stg2))
        stg4 = self.mod4(stg3)
        stg4 = self.mod5(stg4)
        stg4 = self.mod6(stg4)
        stg4 = self.mod7(stg4)
        stg4 = self.bn_out(stg4)
        if self.classes != 0:
            de_stg1 = self.aspp(stg4)[1]
            net_out = self.score(torch.cat([de_stg1, self.stg3_fusion(stg3)], dim=1))
            return net_out
        else:
            return stg4


class ShuffleNetV2Plus(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, n_class=19, groups=3, in_channels=3, in_size=(448, 896), out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 19 for ImageNet.

        """
        super(ShuffleNetV2Plus, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.n_class = n_class
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError("""{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2, dilate=2)
        self.stage3 = self._make_stage(3, dilate=4)
        self.stage4 = self._make_stage(4, dilate=8)
        self.last_channel = 2 * self.stage_out_channels[1] + self.stage_out_channels[2] + self.stage_out_channels[3] + self.stage_out_channels[4]
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
        if self.n_class != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, out_sec, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)), aspp_sec=aspp_sec, norm_act=norm_act))
            in_stag2_up_chs = 2 * self.stage_out_channels[1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([('norm.1', norm_act(out_sec + in_stag2_up_chs)), ('conv.1', nn.Conv2d(out_sec + in_stag2_up_chs, out_sec + in_stag2_up_chs, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)), ('norm.2', norm_act(out_sec + in_stag2_up_chs)), ('conv.2', nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class, kernel_size=1, stride=1, padding=0, bias=True)), ('up1', nn.Upsample(size=in_size, mode='bilinear'))]))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_stage(self, stage, dilate=1):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)
        grouped_conv = stage > 2
        if stage >= 3:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self.stage_out_channels[stage], groups=self.groups, dilate=dilate, grouped_conv=grouped_conv, combine='concat', up=True)
        else:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self.stage_out_channels[stage], groups=self.groups, dilate=1, grouped_conv=grouped_conv, combine='concat', up=False)
        modules[stage_name + '_0'] = first_module
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + '_{}'.format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.stage_out_channels[stage], groups=self.groups, dilate=dilate, grouped_conv=True, combine='add', up=False)
            modules[name] = module
        return nn.Sequential(modules)

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        stg0 = self.conv1(x)
        stg1 = self.maxpool(stg0)
        stg2 = self.stage2(stg1)
        stg3 = self.stage3(stg2)
        stg4 = self.stage4(stg3)
        stg1_1 = F.avg_pool2d(input=stg0, kernel_size=3, stride=2, padding=1)
        stg1_2 = F.avg_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)
        stg1_3 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)
        stg5 = self.out_se(torch.cat([stg2, stg3, stg4, stg1_2, stg1_3], dim=1))
        if self.n_class != 0:
            de_stg1 = self.aspp(stg5)[1]
            de_stg1 = self.score_se(torch.cat([de_stg1, stg1, stg1_1], dim=1))
            net_out = self.score(de_stg1)
            return net_out
        else:
            return stg5


class InPlaceABNSync(autograd.Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var, extra, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        cls._parse_extra(ctx, extra)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        n = _count_samples(x) * (ctx.master_queue.maxsize + 1)
        if ctx.training:
            mean = x.new().resize_(1, running_mean.size(0))
            var = x.new().resize_(1, running_var.size(0))
            _check_contiguous(x, mean, var)
            _ext.bn_mean_var_cuda(x, mean, var)
            if ctx.is_master:
                means, vars = [mean], [var]
                for _ in range(ctx.master_queue.maxsize):
                    mean_w, var_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    means.append(mean_w)
                    vars.append(var_w)
                means = comm.gather(means)
                vars = comm.gather(vars)
                mean = means.mean(0)
                var = (vars + (mean - means) ** 2).mean(0)
                tensors = comm.broadcast_coalesced((mean, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((mean, var))
                mean, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * n / (n - 1))
        else:
            mean, var = running_mean, running_var
        _check_contiguous(x, mean, var, weight, bias)
        _ext.bn_forward_cuda(x, mean, var, weight if weight is not None else x.new(), bias if bias is not None else x.new(), x, x, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, weight, bias, running_mean, running_var)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, weight, bias, running_mean, running_var = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz)
        else:
            dx = None
        if ctx.needs_input_grad[1]:
            dweight = dz.new().resize_as_(running_mean).zero_()
        else:
            dweight = None
        if ctx.needs_input_grad[2]:
            dbias = dz.new().resize_as_(running_mean).zero_()
        else:
            dbias = None
        if ctx.training:
            edz = dz.new().resize_as_(running_mean)
            eydz = dz.new().resize_as_(running_mean)
            _check_contiguous(z, dz, weight, bias, edz, eydz)
            _ext.bn_edz_eydz_cuda(z, dz, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz)
            if ctx.is_master:
                edzs, eydzs = [edz], [eydz]
                for _ in range(len(ctx.worker_queues)):
                    edz_w, eydz_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    edzs.append(edz_w)
                    eydzs.append(eydz_w)
                edz = comm.reduce_add(edzs) / (ctx.master_queue.maxsize + 1)
                eydz = comm.reduce_add(eydzs) / (ctx.master_queue.maxsize + 1)
                tensors = comm.broadcast_coalesced((edz, eydz), [edz.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((edz, eydz))
                edz, eydz = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
        else:
            edz = dz.new().resize_as_(running_mean).zero_()
            eydz = dz.new().resize_as_(running_mean).zero_()
        _check_contiguous(dz, z, ctx.var, weight, bias, edz, eydz, dx, dweight, dbias)
        _ext.bn_backard_cuda(dz, z, ctx.var, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, dx if dx is not None else dz.new(), dweight if dweight is not None else dz.new(), dbias if dbias is not None else dz.new(), ctx.eps)
        del ctx.var
        return dx, dweight, dbias, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra['is_master']
        if ctx.is_master:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queues = extra['worker_queues']
            ctx.worker_ids = extra['worker_ids']
        else:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queue = extra['worker_queue']


class InPlaceABNWrapper(nn.Module):
    """Wrapper module to make `InPlaceABN` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNWrapper, self).__init__()
        self.bn = InPlaceABN(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class InPlaceABNSyncWrapper(nn.Module):
    """Wrapper module to make `InPlaceABNSync` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNSyncWrapper, self).__init__()
        self.bn = InPlaceABNSync(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class ContextEncodeInplaceABN(nn.Module):

    def __init__(self, channel, K=16, reduction=4, norm_act=InPlaceABN):
        super(ContextEncodeInplaceABN, self).__init__()
        out_channel = int(channel / reduction)
        self.pre_abn = norm_act(channel)
        self.context_enc = nn.Sequential(norm_act(channel), nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0), norm_act(out_channel), encoding.nn.Encoding(D=out_channel, K=K), encoding.nn.View(-1, out_channel * K), encoding.nn.Normalize())
        self.channel_se = nn.Sequential(nn.Linear(out_channel * K, channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        pre_x = self.pre_abn(x.clone())
        encode = self.context_enc(pre_x)
        chn_se = self.channel_se(encode).view(batch_size, num_channels, 1, 1)
        spa_se = self.spatial_se(pre_x)
        return encode, torch.mul(torch.mul(x, spa_se), chn_se)


class DPDenseModule(nn.Module):

    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers, dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DPDenseModule, self).__init__()
        self.n_layers = n_layers
        self.convs3 = nn.ModuleList()
        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns
            mid_out = int(self.last_channel * squeeze_ratio)
            self.convs3.append(nn.Sequential(OrderedDict([('bn.1', norm_act(self.last_channel)), ('conv_up', nn.Conv2d(self.last_channel, mid_out, kernel_size=1, stride=1, padding=0, bias=False)), ('bn.2', norm_act(mid_out)), ('dconv', nn.Conv2d(mid_out, mid_out, kernel_size=3, stride=1, padding=dilate, groups=mid_out, dilation=dilate, bias=False)), ('pconv', nn.Conv2d(mid_out, out_chns, kernel_size=1, stride=1, padding=0, bias=False)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))])))
            """
            self.convs3.append(nn.Sequential(OrderedDict([("bn.1", norm_act(self.last_channel)),
                                                          ("dconv", nn.Conv2d(self.last_channel, self.last_channel,
                                                                              kernel_size=3, stride=1, padding=dilate,
                                                                              groups=self.last_channel, dilation=dilate,
                                                                              bias=False)),
                                                          ("pconv", nn.Conv2d(self.last_channel, out_chns,
                                                                              kernel_size=1, stride=1, padding=0,
                                                                              bias=False)),
                                                          ("dropout", nn.Dropout2d(p=0.2, inplace=True))])))
            """

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class GroupNorm2D(nn.Module):
    """
        Group Normalization
        Reference: https://128.84.21.199/abs/1803.08494v1
    """

    def __init__(self, num_features, num_groups=32, eps=1e-05):
        """

        :param num_features:
        :param num_groups:
        :param eps:
        """
        super(GroupNorm2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.num_groups == 0
        x = x.view(batch_size, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(batch_size, num_channels, height, width)
        return x * self.weight + self.bias


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return inputs.view((in_size[0], in_size[1], 1, 1))


class LightHeadBlock(nn.Module):

    def __init__(self, in_chs, mid_chs=256, out_chs=256, kernel_size=15, norm_act=ABN):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.abn = norm_act(in_chs)
        self.conv_l = nn.Sequential(OrderedDict([('conv_lu', nn.Conv2d(in_chs, mid_chs, kernel_size=(kernel_size, 1), padding=(pad, 0))), ('conv_ld', nn.Conv2d(mid_chs, out_chs, kernel_size=(1, kernel_size), padding=(0, pad)))]))
        self.conv_r = nn.Sequential(OrderedDict([('conv_ru', nn.Conv2d(in_chs, mid_chs, kernel_size=(1, kernel_size), padding=(0, pad))), ('conv_rd', nn.Conv2d(mid_chs, out_chs, kernel_size=(kernel_size, 1), padding=(pad, 0)))]))

    def forward(self, x):
        x = self.abn(x)
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


class ASPPBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, aspp_sec=(12, 24, 36)):
        super(ASPPBlock, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode='bilinear')), ('bn0', nn.BatchNorm2d(num_features=out_chs))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1)), ('bn1_1', nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv2_1', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[0], bias=False, groups=1, dilation=aspp_sec[0])), ('bn2_1', nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv2_2', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[1], bias=False, groups=1, dilation=aspp_sec[1])), ('bn2_2', nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv2_3', nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[2], bias=False, groups=1, dilation=aspp_sec[2])), ('bn2_3', nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('conv_down', nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups=1, dilation=1)), ('bn_down', nn.BatchNorm2d(num_features=out_chs)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        out = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1(x), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        out = self.aspp_catdown(out)
        return self.upsampling(out)


class SCSEInvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(SCSEInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True), SCSEBlock(channel=oup, reduction=2))

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """

    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25, gamma=2, size_average=True):
        """
        Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        :param num_classes:   (int) num of the classes
        :param ignore_label:  (int) ignore label
        :param alpha:         (1D Tensor or Variable) the scalar factor
        :param gamma:         (float) gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        :param size_average:  (bool): By default, the losses are averaged over observations for each mini-batch.
                                      If the size_average is set to False, the losses are
                                      instead summed for each mini-batch.
        """
        super(FocalLoss2D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.size_average = size_average
        self.one_hot = Variable(torch.eye(self.num_classes))

    def forward(self, cls_preds, cls_targets):
        """

        :param cls_preds:    (n, c, h, w)
        :param cls_targets:  (n, h, w)
        :return:
        """
        assert not cls_targets.requires_grad
        assert cls_targets.dim() == 3
        assert cls_preds.size(0) == cls_targets.size(0), '{0} vs {1} '.format(cls_preds.size(0), cls_targets.size(0))
        assert cls_preds.size(2) == cls_targets.size(1), '{0} vs {1} '.format(cls_preds.size(2), cls_targets.size(1))
        assert cls_preds.size(3) == cls_targets.size(2), '{0} vs {1} '.format(cls_preds.size(3), cls_targets.size(3))
        if cls_preds.is_cuda:
            self.one_hot = self.one_hot
        n, c, h, w = cls_preds.size()
        cls_targets = cls_targets.view(n * h * w, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)
        prob = F.softmax(cls_preds, dim=1)
        prob = prob.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        prob = prob[target_mask.repeat(1, c)]
        prob = prob.view(-1, c)
        probs = torch.clamp((prob * cls_targets).sum(1).view(-1, 1), min=1e-08, max=1.0)
        batch_loss = -self.alpha * torch.pow(1 - probs, self.gamma) * probs.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class SemanticEncodingLoss(nn.Module):

    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25):
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def unique_encode(self, cls_targets):
        batch_size, _, _ = cls_targets.size()
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = [cls_targets[idx].masked_select(target_mask[idx]) for idx in np.arange(batch_size)]
        unique_cls = [np.unique(label.numpy()) for label in cls_targets]
        encode = np.zeros((batch_size, self.num_classes), dtype=np.uint8)
        for idx in np.arange(batch_size):
            np.put(encode[idx], unique_cls[idx], 1)
        return torch.from_numpy(encode).float()

    def forward(self, predicts, enc_cls_target, size_average=True):
        se_loss = F.binary_cross_entropy_with_logits(predicts, enc_cls_target, weight=None, size_average=size_average)
        return self.alpha * se_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ABN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPBlock,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 56, 112])], {}),
     True),
    (ASPPInPlaceABNBlock,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 56, 112])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (CatInPlaceABN,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseModule,
     lambda: ([], {'in_chns': 4, 'squeeze_ratio': 4, 'out_chns': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityResidualBlock,
     lambda: ([], {'in_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'dilate': 4, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LightHeadBlock,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (ModifiedSCSEBlock,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SCSEBlock,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SCSEInvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'dilate': 4, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SDASPPInPlaceABNBlock,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 56, 112])], {}),
     True),
    (SEBlock,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SemanticEncodingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SemanticSupervision,
     lambda: ([], {'in_chns': 4, 'out_chns': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VortexPooling,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 56, 112])], {}),
     True),
]

class Test_ansleliu_LightNet(_paritybench_base):
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

