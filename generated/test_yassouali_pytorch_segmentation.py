import sys
_module = sys.modules[__name__]
del sys
base = _module
base_dataloader = _module
base_dataset = _module
base_model = _module
base_trainer = _module
dataloaders = _module
ade20k = _module
cityscapes = _module
coco = _module
voc = _module
inference = _module
models = _module
deeplabv3_plus = _module
duc_hdc = _module
enet = _module
fcn = _module
gcn = _module
pspnet = _module
resnet = _module
segnet = _module
unet = _module
upernet = _module
train = _module
trainer = _module
utils = _module
helpers = _module
logger = _module
losses = _module
lovasz_losses = _module
lr_scheduler = _module
metrics = _module
palette = _module
sync_batchnorm = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
torchsummary = _module
transforms = _module

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


from copy import deepcopy


import torch


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


import random


from torch.utils.data import Dataset


from torchvision import transforms


from scipy import ndimage


import logging


import torch.nn as nn


import math


from torch.utils import tensorboard


import scipy.io as sio


import scipy


import torch.nn.functional as F


from math import ceil


from collections import OrderedDict


from torchvision import models


import torch.utils.model_zoo as model_zoo


from itertools import chain


import collections


import torchvision


from torch import nn


import inspect


import time


from torchvision.utils import make_grid


from sklearn.utils import class_weight


from torch.autograd import Variable


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


import functools


from torch.nn.parallel.data_parallel import DataParallel


import numbers


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0001)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class ResNet(nn.Module):

    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if output_stride == 16:
            s3, s4, d3, d4 = 2, 1, 1, 2
        elif output_stride == 8:
            s3, s4, d3, d4 = 1, 1, 2, 4
        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = s3, s3
        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = s4, s4

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


_BATCH_NORM_PARAMS = {'eps': 0.001, 'momentum': 0.9997, 'affine': True}


def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d 
            operation. Should be a positive integer.
        rate: An integer, rate for atrous convolution.
        
    Returns:
        padded_inputs: A tensor of size [batch, height_out, width_out, 
            channels] with the input, either intact (if kernel_size == 1) or 
            padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = torch.nn.functional.pad(inputs, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Conv2dSame(torch.nn.Module):
    """Strided 2-D convolution with 'SAME' padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, rate=1):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
        """
        super(Conv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1
        if self._without_padding:
            padding = (kernel_size - 1) * rate // 2
            self._conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=rate, padding=padding, bias=False)
        else:
            self._conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=rate, bias=False)
        self._batch_norm = torch.nn.BatchNorm2d(out_channels, **_BATCH_NORM_PARAMS)
        self._relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class SeparableConv2dSame(torch.nn.Module):
    """Strided 2-D separable convolution with 'SAME' padding."""

    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier, stride, rate, use_explicit_padding=True, activation_fn=None, regularize_depthwise=False, **kwargs):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            depth_multiplier: The number of depthwise convolution output
                channels for each input channel. The total number of depthwise
                convolution output channels will be equal to `num_filters_in *
                depth_multiplier`.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
            activation_fn: Activation function.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            **kwargs: Additional keyword arguments to pass to torch.nn.Conv2d.
        """
        super(SeparableConv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1 or not use_explicit_padding
        out_channels_depthwise = in_channels * depth_multiplier
        if self._without_padding:
            padding = (kernel_size - 1) * rate // 2
            self._conv_depthwise = torch.nn.Conv2d(in_channels, out_channels_depthwise, kernel_size=kernel_size, stride=stride, dilation=rate, groups=in_channels, padding=padding, bias=False, **kwargs)
        else:
            self._conv_depthwise = torch.nn.Conv2d(in_channels, out_channels_depthwise, kernel_size=kernel_size, stride=stride, dilation=rate, groups=in_channels, bias=False, **kwargs)
        self._batch_norm_depthwise = torch.nn.BatchNorm2d(out_channels_depthwise, **_BATCH_NORM_PARAMS)
        self._conv_pointwise = torch.nn.Conv2d(out_channels_depthwise, out_channels, kernel_size=1, stride=1, bias=False, **kwargs)
        self._batch_norm_pointwise = torch.nn.BatchNorm2d(out_channels, **_BATCH_NORM_PARAMS)
        self._activation_fn = activation_fn

    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv_depthwise(x)
        x = self._batch_norm_depthwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        x = self._conv_pointwise(x)
        x = self._batch_norm_pointwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


_CLIP_CAP = 6


class XceptionModule(torch.nn.Module):
    """An Xception module.
    
    The output of one Xception module is equal to the sum of `residual` and
    `shortcut`, where `residual` is the feature computed by three seperable
    convolution. The `shortcut` is the feature computed by 1x1 convolution
    with or without striding. In some cases, the `shortcut` path could be a
    simple identity function or none (i.e, no shortcut).
    """

    def __init__(self, in_channels, depth_list, skip_connection_type, stride, unit_rate_list, rate=1, activation_fn_in_separable_conv=False, regularize_depthwise=False, use_bounded_activation=False, use_explicit_padding=True):
        """Constructor.
        
        Args:
            in_channels: An integer, the number of input filters.
            depth_list: A list of three integers specifying the depth values
                of one Xception module.
            skip_connection_type: Skip connection type for the residual path.
                Only supports 'conv', 'sum', or 'none'.
            stride: The block unit's stride. Detemines the amount of 
                downsampling of the units output compared to its input.
            unit_rate_list: A list of three integers, determining the unit 
                rate for each separable convolution in the Xception module.
            rate: An integer, rate for atrous convolution.
            activation_fn_in_separable_conv: Includes activation function in
                the seperable convolution or not.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            use_bounded_activation: Whether or not to use bounded activations.
                Bounded activations better lend themselves to quantized 
                inference.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
                
        Raises:
            ValueError: If depth_list and unit_rate_list do not contain three
                integers, or if stride != 1 for the third seperable convolution
                operation in the residual path, or unsupported skip connection
                type.
        """
        super(XceptionModule, self).__init__()
        if len(depth_list) != 3:
            raise ValueError('Expect three elements in `depth_list`.')
        if len(unit_rate_list) != 3:
            raise ValueError('Expect three elements in `unit_rate_list`.')
        if skip_connection_type not in ['conv', 'sum', 'none']:
            raise ValueError('Unsupported skip connection type.')
        self._input_activation_fn = None
        if activation_fn_in_separable_conv:
            activation_fn = torch.nn.ReLU6(inplace=False) if use_bounded_activation else torch.nn.ReLU(inplace=False)
        elif use_bounded_activation:
            activation_fn = lambda x: torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
            self._input_activation_fn = torch.nn.ReLU6(inplace=False)
        else:
            activation_fn = None
            self._input_activation_fn = torch.nn.ReLU(inplace=False)
        self._use_bounded_activation = use_bounded_activation
        self._output_activation_fn = None
        if use_bounded_activation:
            self._output_activation_fn = torch.nn.ReLU6(inplace=True)
        layers = []
        in_channels_ = in_channels
        for i in range(3):
            if self._input_activation_fn is not None:
                layers += [self._input_activation_fn]
            layers += [SeparableConv2dSame(in_channels_, depth_list[i], kernel_size=3, depth_multiplier=1, regularize_depthwise=regularize_depthwise, rate=rate * unit_rate_list[i], stride=stride if i == 2 else 1, activation_fn=activation_fn, use_explicit_padding=use_explicit_padding)]
            in_channels_ = depth_list[i]
        self._separable_conv_block = torch.nn.Sequential(*layers)
        self._skip_connection_type = skip_connection_type
        if skip_connection_type == 'conv':
            self._conv_skip_connection = torch.nn.Conv2d(in_channels, depth_list[-1], kernel_size=1, stride=stride)
            self._batch_norm_shortcut = torch.nn.BatchNorm2d(depth_list[-1], **_BATCH_NORM_PARAMS)

    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height, width, channels].
        
        Returns:
            The Xception module's output.
        """
        residual = self._separable_conv_block(x)
        if self._skip_connection_type == 'conv':
            shortcut = self._conv_skip_connection(x)
            shortcut = self._batch_norm_shortcut(shortcut)
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                shortcut = torch.clamp(shortcut, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + shortcut
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        elif self._skip_connection_type == 'sum':
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                x = torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + x
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        else:
            outputs = residual
        return outputs


class StackBlocksDense(torch.nn.Module):
    """Stacks Xception blocks and controls output feature density.
    
    This class allows the user to explicitly control the output stride, which
    is the ratio of the input to output spatial resolution. This is useful for
    dense prediction tasks such as semantic segmentation or object detection.
    
    Control of the output feature density is implemented by atrous convolution.
    """

    def __init__(self, blocks, output_stride=None):
        """Constructor.
        
        Args:
            blocks: A list of length equal to the number of Xception blocks.
                Each element is an Xception Block object describing the units
                in the block.
            output_stride: If None, then the output will be computed at the
                nominal network stride. If output_stride is not None, it 
                specifies the requested ratio of input to output spatial
                resolution, which needs to be equal to the product of unit
                strides from the start up to some level of Xception. For
                example, if the Xception employs units with strides 1, 2, 1,
                3, 4, 1, then valid values for the output_stride are 1, 2, 6,
                24 or None (which is equivalent to output_stride=24).
                
        Raises:
            ValueError: If the target output_stride is not valid.
        """
        super(StackBlocksDense, self).__init__()
        current_stride = 1
        rate = 1
        layers = []
        for block in blocks:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
                if output_stride is not None and current_stride == output_stride:
                    layers += [block.unit_fn(rate=rate, **dict(unit, stride=1))]
                    rate *= unit.get('stride', 1)
                else:
                    layers += [block.unit_fn(rate=1, **unit)]
                    current_stride *= unit.get('stride', 1)
        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target ouput_stride cannot be reached.')
        self._blocks = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch, height, widht, channels].
            
        Returns:
            Output tensor with stride equal to the specified output_stride.
        """
        x = self._blocks(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        rep = []
        self.relu = nn.ReLU(inplace=True)
        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [self.relu, SeparableConv2d(in_channels, in_channels, 3, 1, dilation), nn.BatchNorm2d(in_channels)]
        if not use_1st_relu:
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        x = output + skip
        return x


class Xception(nn.Module):

    def __init__(self, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()
        if output_stride == 16:
            b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8:
            b3_s, mf_d, ef_d = 1, 2, (2, 4)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)
        for i in range(16):
            exec(f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)')
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0], exit_flow=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = nn.BatchNorm2d(2048)
        initialize_weights(self)
        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        url = 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
        pretrained_weights = model_zoo.load_url(url)
        state_dict = self.state_dict()
        model_dict = {}
        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
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
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
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
        return x, low_level_features


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_channles), nn.ReLU(inplace=True))


class ASSP(nn.Module):

    def __init__(self, in_channels, output_stride, assp_channels=6):
        super(ASSP, self).__init__()
        assert output_stride in [4, 8], 'Only output strides of 8 or 16 are suported'
        assert assp_channels in [4, 6], 'Number of suported ASSP branches are 4 or 6'
        dilations = [1, 6, 12, 18, 24, 36]
        dilations = dilations[:assp_channels]
        self.assp_channels = assp_channels
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])
        if self.assp_channels == 6:
            self.aspp5 = assp_branch(in_channels, 256, 3, dilation=dilations[4])
            self.aspp6 = assp_branch(in_channels, 256, 3, dilation=dilations[5])
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(256 * (self.assp_channels + 1), 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        if self.assp_channels == 6:
            x5 = self.aspp5(x)
            x6 = self.aspp6(x)
        x_avg_pool = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        if self.assp_channels == 6:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x5, x6, x_avg_pool), dim=1))
        else:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x_avg_pool), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        return x


class DUC(nn.Module):

    def __init__(self, in_channels, out_channles, upscale):
        super(DUC, self).__init__()
        out_channles = out_channles * upscale ** 2
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)
        initialize_weights(self)
        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        """
        Even with pixel shuffle we still have check board artifacts,
        the solution is to initialize the d**2 feature maps with the same
        radom weights: https://arxiv.org/pdf/1707.02937.pdf
        """
        new_shape = [int(x.shape[0] / scale ** 2)] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel


class Decoder(nn.Module):

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.DUC = DUC(256, 256, upscale=2)
        self.output = nn.Sequential(nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(256, num_classes, 1, stride=1))
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        x = self.DUC(x)
        if x.size() != low_level_features.size():
            x = x[:, :, :low_level_features.size(2), :low_level_features.size(3)]
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


class DeepLab(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True, output_stride=16, freeze_bn=False, freeze_backbone=False, **_):
        super(DeepLab, self).__init__()
        assert 'xception' or 'resnet' in backbone
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 128
        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class ResNet_HDC_DUC(nn.Module):

    def __init__(self, in_channels, output_stride, pretrained=True, dilation_bigger=False):
        super(ResNet_HDC_DUC, self).__init__()
        model = models.resnet101(pretrained=pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if output_stride == 4:
            list(self.layer0.children())[0].stride = 1, 1
        d_res4b = []
        if dilation_bigger:
            d_res4b.extend([1, 2, 5, 9] * 5 + [1, 2, 5])
            d_res5b = [5, 9, 17]
        else:
            d_res4b.extend([1, 2, 3] * 7 + [2, 2])
            d_res5b = [3, 4, 5]
        l_index = 0
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                d = d_res4b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        l_index = 0
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                d = d_res5b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = 1, 1

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_features


class DeepLab_DUC_HDC(BaseModel):

    def __init__(self, num_classes, in_channels=3, pretrained=True, output_stride=8, freeze_bn=False, **_):
        super(DeepLab_DUC_HDC, self).__init__()
        self.backbone = ResNet_HDC_DUC(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
        low_level_channels = 256
        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)
        self.DUC_out = DUC(num_classes, num_classes, 4)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = self.DUC_out(x)
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters(), self.DUC_out.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class InitalBlock(nn.Module):

    def __init__(self, in_channels, use_prelu=True):
        super(InitalBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16) if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels=None, activation=None, dilation=1, downsample=False, proj_ratio=4, upsample=False, asymetric=False, regularize=True, p_drop=None, use_prelu=True):
        super(BottleNeck, self).__init__()
        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        if out_channels is None:
            out_channels = in_channels
        else:
            self.pad = out_channels - in_channels
        if regularize:
            assert p_drop is not None
        if downsample:
            assert not upsample
        elif upsample:
            assert not downsample
        inter_channels = in_channels // proj_ratio
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        if asymetric:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 5), padding=(0, 2)), nn.BatchNorm2d(inter_channels), nn.PReLU() if use_prelu else nn.ReLU(inplace=True), nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), padding=(2, 0)))
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        identity = x
        if self.upsample:
            assert indices is not None and output_size is not None
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0
                identity = F.pad(identity, pad, 'constant', 0)
            identity = self.unpool(identity, indices=indices)
        elif self.downsample:
            identity, idx = self.pool(identity)
        """
        if self.pad > 0:
            if self.pad % 2 == 0 : pad = (0, 0, 0, 0, self.pad//2, self.pad//2)
            else: pad = (0, 0, 0, 0, self.pad//2, self.pad//2+1)
            identity = F.pad(identity, pad, "constant", 0)
        """
        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
            if torch.cuda.is_available():
                extras = extras
            identity = torch.cat((identity, extras), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)
        if identity.size() != x.size():
            pad = identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0
            x = F.pad(x, pad, 'constant', 0)
        x += identity
        x = self.prelu_out(x)
        if self.downsample:
            return x, idx
        return x


class ENet(BaseModel):

    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ENet, self).__init__()
        self.initial = InitalBlock(in_channels)
        self.bottleneck10 = BottleNeck(16, 64, downsample=True, p_drop=0.01)
        self.bottleneck11 = BottleNeck(64, p_drop=0.01)
        self.bottleneck12 = BottleNeck(64, p_drop=0.01)
        self.bottleneck13 = BottleNeck(64, p_drop=0.01)
        self.bottleneck14 = BottleNeck(64, p_drop=0.01)
        self.bottleneck20 = BottleNeck(64, 128, downsample=True, p_drop=0.1)
        self.bottleneck21 = BottleNeck(128, p_drop=0.1)
        self.bottleneck22 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck23 = BottleNeck(128, asymetric=True, p_drop=0.1)
        self.bottleneck24 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck25 = BottleNeck(128, p_drop=0.1)
        self.bottleneck26 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck27 = BottleNeck(128, asymetric=True, p_drop=0.1)
        self.bottleneck28 = BottleNeck(128, dilation=16, p_drop=0.1)
        self.bottleneck31 = BottleNeck(128, p_drop=0.1)
        self.bottleneck32 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck33 = BottleNeck(128, asymetric=True, p_drop=0.1)
        self.bottleneck34 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck35 = BottleNeck(128, p_drop=0.1)
        self.bottleneck36 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck37 = BottleNeck(128, asymetric=True, p_drop=0.1)
        self.bottleneck38 = BottleNeck(128, dilation=16, p_drop=0.1)
        self.bottleneck40 = BottleNeck(128, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck41 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        self.bottleneck42 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        self.bottleneck50 = BottleNeck(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck51 = BottleNeck(16, p_drop=0.1, use_prelu=False)
        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        initialize_weights(self)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        x = self.initial(x)
        sz1 = x.size()
        x, indices1 = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        sz2 = x.size()
        x, indices2 = self.bottleneck20(x)
        x = self.bottleneck21(x)
        x = self.bottleneck22(x)
        x = self.bottleneck23(x)
        x = self.bottleneck24(x)
        x = self.bottleneck25(x)
        x = self.bottleneck26(x)
        x = self.bottleneck27(x)
        x = self.bottleneck28(x)
        x = self.bottleneck31(x)
        x = self.bottleneck32(x)
        x = self.bottleneck33(x)
        x = self.bottleneck34(x)
        x = self.bottleneck35(x)
        x = self.bottleneck36(x)
        x = self.bottleneck37(x)
        x = self.bottleneck38(x)
        x = self.bottleneck40(x, indices=indices2, output_size=sz2)
        x = self.bottleneck41(x)
        x = self.bottleneck42(x)
        x = self.bottleneck50(x, indices=indices1, output_size=sz1)
        x = self.bottleneck51(x)
        x = self.fullconv(x)
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8(BaseModel):

    def __init__(self, num_classes, pretrained=True, freeze_bn=False, **_):
        super(FCN8, self).__init__()
        vgg = models.vgg16(pretrained)
        features = list(vgg.features.children())
        classifier = list(vgg.classifier.children())
        features[0].padding = 100, 100
        for layer in features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])
        self.adj_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.adj_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        output = nn.Conv2d(4096, num_classes, kernel_size=1)
        conv6.weight.data.copy_(classifier[0].weight.data.view(conv6.weight.data.size()))
        conv6.bias.data.copy_(classifier[0].bias.data)
        conv7.weight.data.copy_(classifier[3].weight.data.view(conv7.weight.data.size()))
        conv7.bias.data.copy_(classifier[3].bias.data)
        self.output = nn.Sequential(conv6, nn.ReLU(inplace=True), nn.Dropout(), conv7, nn.ReLU(inplace=True), nn.Dropout(), output)
        self.up_output = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.up_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.up_output.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.up_pool4_out.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.up_final.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.requires_grad = False
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.pool3, self.pool4, self.pool5], False)

    def forward(self, x):
        imh_H, img_W = x.size()[2], x.size()[3]
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        output = self.output(pool5)
        up_output = self.up_output(output)
        adjstd_pool4 = self.adj_pool4(0.01 * pool4)
        add_out_pool4 = self.up_pool4_out(adjstd_pool4[:, :, 5:5 + up_output.size()[2], 5:5 + up_output.size()[3]] + up_output)
        adjstd_pool3 = self.adj_pool3(0.0001 * pool3)
        final_value = self.up_final(adjstd_pool3[:, :, 9:9 + add_out_pool4.size()[2], 9:9 + add_out_pool4.size()[3]] + add_out_pool4)
        final_value = final_value[:, :, 31:31 + imh_H, 31:31 + img_W].contiguous()
        return final_value

    def get_backbone_params(self):
        return chain(self.pool3.parameters(), self.pool4.parameters(), self.pool5.parameters(), self.output.parameters())

    def get_decoder_params(self):
        return chain(self.up_output.parameters(), self.adj_pool4.parameters(), self.up_pool4_out.parameters(), self.adj_pool3.parameters(), self.up_final.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class Block_Resnet_GCN(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)
        x = x1 + x2
        return x


class BottleneckGCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None
        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.gcn(x)
        x = self.conv1x1(x)
        x = self.bn1x1(x)
        x += identity
        return x


class ResnetGCN(nn.Module):

    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), kernel_sizes=(5, 7)):
        super(ResnetGCN, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=False)
        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = nn.Sequential(BottleneckGCN(512, 1024, kernel_sizes[0], out_channels_gcn[0], stride=2), *([BottleneckGCN(1024, 1024, kernel_sizes[0], out_channels_gcn[0])] * 5))
        self.layer4 = nn.Sequential(BottleneckGCN(1024, 2048, kernel_sizes[1], out_channels_gcn[1], stride=2), *([BottleneckGCN(1024, 1024, kernel_sizes[1], out_channels_gcn[1])] * 5))
        initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = x.size(2), x.size(3)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz


class Resnet(nn.Module):

    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), pretrained=True, kernel_sizes=(5, 7)):
        super(Resnet, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained)
        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if not pretrained:
            initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = x.size(2), x.size(3)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz


class GCN_Block(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels):
        super(GCN_Block, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x = x1 + x2
        return x


class BR_Block(nn.Module):

    def __init__(self, num_channels):
        super(BR_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        initialize_weights(self)

    def forward(self, x):
        identity = x
        x = self.conv2(self.relu2(self.conv1(x)))
        x += identity
        return x


class GCN(BaseModel):

    def __init__(self, num_classes, in_channels=3, pretrained=True, use_resnet_gcn=False, backbone='resnet50', use_deconv=False, num_filters=11, freeze_bn=False, **_):
        super(GCN, self).__init__()
        self.use_deconv = use_deconv
        if use_resnet_gcn:
            self.backbone = ResnetGCN(in_channels, backbone=backbone)
        else:
            self.backbone = Resnet(in_channels, pretrained=pretrained, backbone=backbone)
        if backbone == 'resnet34' or backbone == 'resnet18':
            resnet_channels = [64, 128, 256, 512]
        else:
            resnet_channels = [256, 512, 1024, 2048]
        self.gcn1 = GCN_Block(num_filters, resnet_channels[0], num_classes)
        self.br1 = BR_Block(num_classes)
        self.gcn2 = GCN_Block(num_filters, resnet_channels[1], num_classes)
        self.br2 = BR_Block(num_classes)
        self.gcn3 = GCN_Block(num_filters, resnet_channels[2], num_classes)
        self.br3 = BR_Block(num_classes)
        self.gcn4 = GCN_Block(num_filters, resnet_channels[3], num_classes)
        self.br4 = BR_Block(num_classes)
        self.br5 = BR_Block(num_classes)
        self.br6 = BR_Block(num_classes)
        self.br7 = BR_Block(num_classes)
        self.br8 = BR_Block(num_classes)
        self.br9 = BR_Block(num_classes)
        if self.use_deconv:
            self.decon1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
            self.decon5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        x1, x2, x3, x4, conv1_sz = self.backbone(x)
        x1 = self.br1(self.gcn1(x1))
        x2 = self.br2(self.gcn2(x2))
        x3 = self.br3(self.gcn3(x3))
        x4 = self.br4(self.gcn4(x4))
        if self.use_deconv:
            x4 = self.decon4(x4)
            if x4.size() != x3.size():
                x4 = self._pad(x4, x3)
            x3 = self.decon3(self.br5(x3 + x4))
            if x3.size() != x2.size():
                x3 = self._pad(x3, x2)
            x2 = self.decon2(self.br6(x2 + x3))
            x1 = self.decon1(self.br7(x1 + x2))
            x = self.br9(self.decon5(self.br8(x1)))
        else:
            x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = F.interpolate(self.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.interpolate(self.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = F.interpolate(self.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)
            x = self.br9(F.interpolate(self.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
        return self.final_conv(x)

    def _pad(self, x_topad, x):
        pad = x.size(3) - x_topad.size(3), 0, x.size(2) - x_topad.size(2), 0
        x_topad = F.pad(x_topad, pad, 'constant', 0)
        return x_topad

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return [p for n, p in self.named_parameters() if 'backbone' not in n]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class _PSPModule(nn.Module):

    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + out_channels * len(bin_sizes), out_channels, kernel_size=3, padding=1, bias=False), norm_layer(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux
        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.master_branch = nn.Sequential(_PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer), nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1))
        self.auxiliary_branch = nn.Sequential(nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False), norm_layer(m_out_sz // 4), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1))
        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = x.size()[2], x.size()[3]
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]
        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class PSPDenseNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='densenet201', pretrained=True, use_aux=True, freeze_bn=False, **_):
        super(PSPDenseNet, self).__init__()
        self.use_aux = use_aux
        model = getattr(models, backbone)(pretrained)
        m_out_sz = model.classifier.in_features
        aux_out_sz = model.features.transition3.conv.out_channels
        if not pretrained or in_channels != 3:
            block0 = [nn.Conv2d(in_channels, 64, 3, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            block0.extend([nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)] * 2)
            self.block0 = nn.Sequential(*block0, nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            initialize_weights(self.block0)
        else:
            self.block0 = nn.Sequential(*list(model.features.children())[:4])
        self.block1 = model.features.denseblock1
        self.block2 = model.features.denseblock2
        self.block3 = model.features.denseblock3
        self.block4 = model.features.denseblock4
        self.transition1 = model.features.transition1
        self.transition2 = nn.Sequential(*list(model.features.transition2.children())[:-1])
        self.transition3 = nn.Sequential(*list(model.features.transition3.children())[:-1])
        for n, m in self.block3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (2, 2), (2, 2)
        for n, m in self.block4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (4, 4), (4, 4)
        self.master_branch = nn.Sequential(_PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d), nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1))
        self.auxiliary_branch = nn.Sequential(nn.Conv2d(aux_out_sz, m_out_sz // 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(m_out_sz // 4), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1))
        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        input_size = x.size()[2], x.size()[3]
        x = self.block0(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        x_aux = self.transition3(x)
        x = self.block4(x_aux)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.block0.parameters(), self.block1.parameters(), self.block2.parameters(), self.block3.parameters(), self.transition1.parameters(), self.transition2.parameters(), self.transition3.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        out += residual
        out = self.relu(out)
        return out


class SegNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i + 3][::-1]]
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i + 1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)
        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:], nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder, self.stage4_decoder, self.stage5_decoder)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)
        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)
        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)
        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)
        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)
        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)
        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)
        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)
        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class DecoderBottleneck(nn.Module):

    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.ConvTranspose2d(inchannels // 4, inchannels // 4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.ConvTranspose2d(inchannels, inchannels // 2, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(inchannels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LastBottleneck(nn.Module):

    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.Conv2d(inchannels // 4, inchannels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False), nn.BatchNorm2d(inchannels // 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SegResNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)
        resnet50_untrained = models.resnet50(pretrained=False)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = 2048, 1024, 512
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))
        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False), nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.first_conv, self.encoder], False)

    def forward(self, x):
        inputsize = x.size()
        x, indices = self.first_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2] - (h_diff - 1), w_diff:x.size()[3] - (w_diff - 1)]
        else:
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff:x.size()[3] - w_diff]
        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)
        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff:x.size()[3] - w_diff]
            if h_diff % 2 != 0:
                x = x[:, :, :-1, :]
            if w_diff % 2 != 0:
                x = x[:, :, :, :-1]
        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(inner_channels), nn.ReLU(inplace=True), nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    return down_conv


class encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3):
            if interpolate:
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode='bilinear', align_corners=True)
            else:
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()
        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)
        self.middle_conv = x2conv(1024, 1024)
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class UNetResnet(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True, freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)
        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        initialize_weights(self)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))
        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))
        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv4(self.conv4(x))
        x = self.upconv5(self.conv5(x))
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(), self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(), self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class PSPModule(nn.Module):

    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + out_channels * len(bin_sizes), in_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):

    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(fpn_out), nn.ReLU(inplace=True))

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]
        x = self.conv_fusion(torch.cat(P, dim=1))
        return x


class UperNet(BaseModel):

    def __init__(self, num_classes, in_channels=3, backbone='resnet101', pretrained=True, use_aux=True, fpn_out=256, freeze_bn=False, **_):
        super(UperNet, self).__init__()
        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        else:
            feature_channels = [256, 512, 1024, 2048]
        self.backbone = ResNet(in_channels, pretrained=pretrained)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        input_size = x.size()[2], x.size()[3]
        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))
        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (2.0 * intersection + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):

    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes is 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


class LovaszSoftmax(nn.Module):

    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASSP,
     lambda: ([], {'in_channels': 4, 'output_stride': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BR_Block,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm2dReimpl,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block_Resnet_GCN,
     lambda: ([], {'kernel_size': 4, 'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'low_level_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBottleneck,
     lambda: ([], {'inchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ENet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (InitalBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastBottleneck,
     lambda: ([], {'inchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LovaszSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PSPDenseNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PSPModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet_HDC_DUC,
     lambda: ([], {'in_channels': 4, 'output_stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'depth_multiplier': 1, 'stride': 1, 'rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_PSPModule,
     lambda: ([], {'in_channels': 4, 'bin_sizes': [4, 4], 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (encoder,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yassouali_pytorch_segmentation(_paritybench_base):
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

