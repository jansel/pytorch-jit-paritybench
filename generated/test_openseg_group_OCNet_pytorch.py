import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
cityscapes = _module
eval = _module
generate_submit = _module
inplace_abn = _module
bn = _module
functions = _module
modules = _module
_ext = _module
bn = _module
build = _module
dense = _module
misc = _module
residual = _module
network = _module
resnet101_asp_oc = _module
resnet101_base_oc = _module
resnet101_baseline = _module
resnet101_pyramid_oc = _module
oc_module = _module
asp_oc_block = _module
base_oc_block = _module
pyramid_oc_block = _module
train = _module
utils = _module
criterion = _module
files = _module
loss = _module
metric = _module
operator = _module
parallel = _module
resnet_block = _module

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


import scipy


from scipy import ndimage


import torch


import numpy as np


import numpy.ma as ma


from torch.autograd import Variable


import torch.nn.functional as F


from torch.utils import data


from collections import OrderedDict


import scipy.ndimage as nd


from math import ceil


import torch.nn as nn


import torch.nn.functional as functional


from collections import Iterable


from itertools import repeat


import torch.autograd as autograd


from torch.nn import functional as F


import math


import torch.utils.model_zoo as model_zoo


import functools


from torch import nn


import torch.optim as optim


import scipy.misc


import torch.backends.cudnn as cudnn


import random


import logging


from torch.autograd import Function


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


ACT_ELU = 'elu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
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
        x = functional.batch_norm(x, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope,
                inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs
        ):
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
        super(ABN, self).__init__(OrderedDict([('bn', nn.BatchNorm2d(
            num_features, **kwargs)), ('act', activation)]))


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__(num_features, eps, momentum,
            affine, activation, slope)

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, self.running_mean,
            self.running_var, self.training, self.momentum, self.eps, self.
            activation, self.slope)


inplace_abn = InPlaceABN.apply


class InPlaceABN(nn.Module):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, autograd.Variable(
            self.running_mean), autograd.Variable(self.running_var), self.
            training, self.momentum, self.eps, self.activation, self.slope)

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization

    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
    """

    def __init__(self, num_features, devices=None, eps=1e-05, momentum=0.1,
        affine=True, activation='leaky_relu', slope=0.01):
        """Creates a synchronized, InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        devices : list of int or None
            IDs of the GPUs that will run the replicas of this module.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABNSync, self).__init__(num_features, eps, momentum,
            affine, activation, slope)
        self.devices = devices if devices else list(range(torch.cuda.
            device_count()))
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        if x.get_device() == self.devices[0]:
            extra = {'is_master': True, 'master_queue': self.master_queue,
                'worker_queues': self.worker_queues, 'worker_ids': self.
                worker_ids}
        else:
            extra = {'is_master': False, 'master_queue': self.master_queue,
                'worker_queue': self.worker_queues[self.worker_ids.index(x.
                get_device())]}
        return inplace_abn_sync(x, self.weight, self.bias, self.
            running_mean, self.running_var, extra, self.training, self.
            momentum, self.eps, self.activation, self.slope)

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, devices={devices}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


inplace_abn_sync = InPlaceABNSync.apply


class InPlaceABNSync(nn.Module):
    """InPlace Activated Batch Normalization with cross-GPU synchronization

    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
    """

    def __init__(self, num_features, devices=None, eps=1e-05, momentum=0.1,
        affine=True, activation='leaky_relu', slope=0.01):
        """Creates a synchronized, InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        devices : list of int or None
            IDs of the GPUs that will run the replicas of this module.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABNSync, self).__init__()
        self.num_features = num_features
        self.devices = devices if devices else list(range(torch.cuda.
            device_count()))
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        if x.get_device() == self.devices[0]:
            extra = {'is_master': True, 'master_queue': self.master_queue,
                'worker_queues': self.worker_queues, 'worker_ids': self.
                worker_ids}
        else:
            extra = {'is_master': False, 'master_queue': self.master_queue,
                'worker_queue': self.worker_queues[self.worker_ids.index(x.
                get_device())]}
        return inplace_abn_sync(x, self.weight, self.bias, autograd.
            Variable(self.running_mean), autograd.Variable(self.running_var
            ), extra, self.training, self.momentum, self.eps, self.
            activation, self.slope)

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, devices={devices}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


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

    def __init__(self, in_channels, growth, layers, bottleneck_factor=4,
        norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([('bn', norm_act(
                in_channels)), ('conv', nn.Conv2d(in_channels, self.growth *
                bottleneck_factor, 1, bias=False))])))
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(
                self.growth * bottleneck_factor)), ('conv', nn.Conv2d(self.
                growth * bottleneck_factor, self.growth, 3, padding=
                dilation, bias=False, dilation=dilation))])))
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

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=
        1, norm_act=ABN, dropout=None):
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
            raise ValueError('channels must contain either two or three values'
                )
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3,
                stride=stride, padding=dilation, bias=False, dilation=
                dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.
                Conv2d(channels[0], channels[1], 3, stride=1, padding=
                dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                stride=stride, padding=0, bias=False)), ('bn2', norm_act(
                channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1],
                3, stride=1, padding=dilation, bias=False, groups=groups,
                dilation=dilation)), ('bn3', norm_act(channels[1])), (
                'conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1,
                padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride
                =stride, padding=0, bias=False)

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


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.context = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3,
            stride=1, padding=1), InPlaceABNSync(512), ASP_OC_Module(512, 256))
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride
            =1, padding=1), InPlaceABNSync(512), nn.Dropout2d(0.1), nn.
            Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
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
        x = self.context(x)
        x = self.cls(x)
        return [x_dsn, x]


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.context = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3,
            stride=1, padding=1), InPlaceABNSync(512), BaseOC_Module(
            in_channels=512, out_channels=512, key_channels=256,
            value_channels=256, dropout=0.05, sizes=[1]))
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride
            =1, padding=1), InPlaceABNSync(512), nn.Dropout2d(0.05), nn.
            Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
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
        x = self.context(x)
        x = self.cls(x)
        return [x_dsn, x]


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.layer5 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3,
            stride=1, padding=1), InPlaceABNSync(512), nn.Dropout2d(0.05))
        self.layer6 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4, multi_grid=(1, 1, 1))
        self.layer5 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3,
            stride=1, padding=1), BatchNorm2d(512))
        self.context = Pyramid_OC_Module(in_channels=512, out_channels=512,
            dropout=0.05, sizes=[1, 2, 3, 6])
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride
            =1, padding=1), BatchNorm2d(512), nn.Dropout2d(0.05), nn.Conv2d
            (512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
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
        x = self.layer5(x)
        x = self.context(x)
        x = self.cls(x)
        return [x_dsn, x]


class ASP_OC_Module(nn.Module):

    def __init__(self, features, out_features=256, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=1, dilation=1, bias=True),
            InPlaceABNSync(out_features), BaseOC_Context_Module(in_channels
            =out_features, out_channels=out_features, key_channels=
            out_features // 2, value_channels=out_features, dropout=0,
            sizes=[2]))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=False), InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=False), InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=False), InPlaceABNSync(out_features))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(out_features * 5, 
            out_features * 2, kernel_size=1, padding=0, dilation=1, bias=
            False), InPlaceABNSync(out_features * 2), nn.Dropout2d(0.1))

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i],
                feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


torch_ver = torch.__version__[:3]


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), InPlaceABNSync(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=
            self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            if torch_ver == '0.4':
                context = F.upsample(input=context, size=(h, w), mode=
                    'bilinear', align_corners=True)
            elif torch_ver == '0.3':
                context = F.upsample(input=context, size=(h, w), mode=
                    'bilinear')
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
            key_channels, value_channels, out_channels, scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, dropout, sizes=[1]):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            out_channels, key_channels, value_channels, size) for size in
            sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels,
            out_channels, kernel_size=1, padding=0), InPlaceABNSync(
            out_channels), nn.Dropout2d(dropout))

    def _make_stage(self, in_channels, output_channels, key_channels,
        value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels,
            value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, dropout, sizes=[1]):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            out_channels, key_channels, value_channels, size) for size in
            sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(in_channels,
            out_channels, kernel_size=1, padding=0), InPlaceABNSync(
            out_channels))

    def _make_stage(self, in_channels, output_channels, key_channels,
        value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels,
            value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class _PyramidSelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), InPlaceABNSync(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=
            self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y +
                    step_w, w)
                if i == self.scale - 1:
                    end_x = h
                if j == self.scale - 1:
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:
                local_y[i + 1]]
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.
                value_channels, -1)
            value_local = value_local.permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.
                key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.
                key_channels, -1)
            sim_map = torch.matmul(query_local, key_local)
            sim_map = self.key_channels ** -0.5 * sim_map
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.
                value_channels, h_local, w_local)
            local_list.append(context_local)
        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(in_channels,
            key_channels, value_channels, out_channels, scale)


class Pyramid_OC_Module(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout, sizes=[1]):
        super(Pyramid_OC_Module, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            out_channels, in_channels // 2, in_channels, size) for size in
            sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels *
            self.group, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels), nn.Dropout2d(dropout))
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels *
            self.group, kernel_size=1, padding=0), InPlaceABNSync(
            in_channels * self.group))

    def _make_stage(self, in_channels, output_channels, key_channels,
        value_channels, size):
        return PyramidSelfAttentionBlock2D(in_channels, key_channels,
            value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output


class CriterionCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 
            0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
            0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
            ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds, size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 
            0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
            0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            None
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_index)
        else:
            None
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN(nn.Module):
    """
    DSN + OHEM : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000,
        dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept,
            use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN_single(nn.Module):
    """
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    """

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000,
        dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 
            0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
            0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
            ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres,
            min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode=
                'bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode=
                'bilinear')
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight = use_weight
        if self.use_weight:
            self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            None
        else:
            self.weight = None

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        if self.use_weight:
            None
            freq = np.zeros(19)
            for k in range(19):
                mask = target[:, :, :] == k
                freq[k] = torch.sum(mask)
                None
            weight = freq / np.sum(freq)
            None
            self.weight = torch.FloatTensor(weight)
            None
        else:
            self.weight = None
        criterion = torch.nn.CrossEntropyLoss(weight=self.weight,
            ignore_index=self.ignore_label)
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)
            ].view(-1, c)
        loss = criterion(predict, target)
        return loss


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight
        =True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            None
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_label)
        else:
            None
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_label)

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
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            None
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        target = Variable(torch.from_numpy(input_label.reshape(target.size(
            ))).long())
        return self.criterion(predict, target)


class Separable_transpose_convolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
        padding=1, output_padding=0, bias=False, dilation=1):
        super(Separable_transpose_convolution, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels,
            kernel_size, stride, padding, output_padding, groups=
            in_channels, bias=bias, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=affine_par)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size):
        x = self.relu(self.bn1(self.conv1(x, output_size)))
        x = self.bn2(self.conv2(x))
        return x


class Separable_convolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=False, dilation=1):
        super(Separable_convolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels, bias=bias, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=affine_par)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


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


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
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
            with torch.cuda.device(device):
                output = module(input, target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
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
        targets = tuple(targets_per_gpu[0] for targets_per_gpu in targets)
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation * multi_grid, dilation=dilation * multi_grid,
            bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_openseg_group_OCNet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(ABN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DenseModule(*[], **{'in_channels': 4, 'growth': 4, 'layers': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(IdentityResidualBlock(*[], **{'in_channels': 4, 'channels': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Separable_convolution(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

