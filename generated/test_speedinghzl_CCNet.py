import sys
_module = sys.modules[__name__]
del sys
cc_attention = _module
_ext = _module
build = _module
functions = _module
dataset = _module
datasets = _module
evaluate = _module
libs = _module
_ext = _module
bn = _module
build = _module
dense = _module
functions = _module
misc = _module
residual = _module
networks = _module
ccnet = _module
deeplabv3 = _module
pspnet = _module
test = _module
train = _module
utils = _module
criterion = _module
encoding = _module
loss = _module
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


import torch


import torch.nn as nn


import torch.autograd as autograd


import torch.cuda.comm as comm


import torch.nn.functional as F


from torch.autograd.function import once_differentiable


import time


import functools


import numpy as np


import random


import collections


import torchvision


from torch.utils import data


import scipy


from scipy import ndimage


from torch.autograd import Variable


import torchvision.models as models


from collections import OrderedDict


import scipy.ndimage as nd


from math import ceil


from collections import Iterable


from itertools import repeat


from torch.nn import functional as F


import math


import torch.utils.model_zoo as model_zoo


import torch.optim as optim


import scipy.misc


import torch.backends.cudnn as cudnn


import logging


from torch.autograd import Function


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


class CA_Map(autograd.Function):

    @staticmethod
    def forward(ctx, weight, g):
        out = torch.zeros_like(g)
        _ext.ca_map_forward_cuda(weight, g, out)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
        _check_contiguous(dw, dg)
        return dw, dg


ca_map = CA_Map.apply


class CA_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        n, c, h, w = t.size()
        size = n, h + w - 1, h, w
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
        _ext.ca_forward_cuda(t, f, weight)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
        _check_contiguous(dt, df)
        return dt, df


ca_weight = CA_Weight.apply


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x
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


ACT_LEAKY_RELU = 'leaky_relu'


ACT_ELU = 'elu'


ACT_NONE = 'none'


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError('CUDA Error encountered in {}'.format(fn))


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_backward_cuda, x, dx, ctx.slope)
        _check(_ext.leaky_relu_cuda, x, 1.0 / ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_backward_cuda, x, dx)
        _check(_ext.elu_inv_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_cuda, x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


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
            _check(_ext.bn_mean_var_cuda, x, mean, var)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * n / (n - 1))
        else:
            mean, var = running_mean, running_var
        _check_contiguous(x, mean, var, weight, bias)
        _check(_ext.bn_forward_cuda, x, mean, var, weight if weight is not None else x.new(), bias if bias is not None else x.new(), x, x, ctx.eps)
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
            _check(_ext.bn_edz_eydz_cuda, z, dz, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, ctx.eps)
        else:
            edz = dz.new().resize_as_(running_mean).zero_()
            eydz = dz.new().resize_as_(running_mean).zero_()
        _check_contiguous(dz, z, ctx.var, weight, bias, edz, eydz, dx, dweight, dbias)
        _check(_ext.bn_backard_cuda, dz, z, ctx.var, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, dx if dx is not None else dz.new(), dweight if dweight is not None else dz.new(), dbias if dbias is not None else dz.new(), ctx.eps)
        del ctx.var
        return dx, dweight, dbias, None, None, None, None, None, None, None


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
            _check(_ext.bn_mean_var_cuda, x, mean, var)
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
        _check(_ext.bn_forward_cuda, x, mean, var, weight if weight is not None else x.new(), bias if bias is not None else x.new(), x, x, ctx.eps)
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
            _check(_ext.bn_edz_eydz_cuda, z, dz, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, ctx.eps)
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
        _check(_ext.bn_backard_cuda, dz, z, ctx.var, weight if weight is not None else dz.new(), bias if bias is not None else dz.new(), edz, eydz, dx if dx is not None else dz.new(), dweight if dweight is not None else dz.new(), dbias if dbias is not None else dz.new(), ctx.eps)
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


class DenseModule(nn.Module):

    def __init__(self, in_channels, growth, layers, bottleneck_factor=4, norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([('bn', norm_act(in_channels)), ('conv', nn.Conv2d(in_channels, self.growth * bottleneck_factor, 1, bias=False))])))
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(self.growth * bottleneck_factor)), ('conv', nn.Conv2d(self.growth * bottleneck_factor, self.growth, 3, padding=dilation, bias=False, dilation=dilation))])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, dropout=None):
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
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
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


BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

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
        out = out + residual
        out = self.relu_inplace(out)
        return out


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False), InPlaceABNSync(out_features), nn.Dropout2d(0.1))

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class RCCAModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False), InPlaceABNSync(out_channels), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.head = nn.Sequential(PSPModule(2048, 512), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), InPlaceABNSync(512), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        return [x, x_dsn]


class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), InPlaceABNSync(inner_features))
        self.bottleneck = nn.Sequential(nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(out_features), nn.Dropout2d(0.1))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        bottle = self.bottleneck(out)
        return bottle


class CriterionCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)
        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)
        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape
        threshold = self.find_threshold(np_predict, np_target)
        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            None
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long()
        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class CriterionOhemCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255, thres=0.6, min_kept=200000):
        super(CriterionOhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return loss1 + loss2 * 0.4


class CriterionOhemDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)
        return loss1 + loss2 * 0.4


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            if not isinstance(input, tuple):
                input = input,
            with torch.device(device):
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input, target, kwargs, device)) for i, (module, input, target, kwargs, device) in enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ABN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CriterionCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4, 4, 4], dtype=torch.int64)], {}),
     True),
    (CriterionOhemCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (DataParallelCriterion,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DataParallelModel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DenseModule,
     lambda: ([], {'in_channels': 4, 'growth': 4, 'layers': 1}),
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
]

class Test_speedinghzl_CCNet(_paritybench_base):
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

