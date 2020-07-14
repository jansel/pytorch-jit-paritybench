import sys
_module = sys.modules[__name__]
del sys
setup = _module
torchfusion = _module
datasets = _module
datasets = _module
fp16_utils = _module
fp16_optimizer = _module
fp16util = _module
loss_scaler = _module
gan = _module
applications = _module
applications = _module
distributions = _module
layers = _module
layers = _module
learners = _module
learners = _module
initializers = _module
initializers = _module
lang = _module
layers = _module
learners = _module
metrics = _module
metrics = _module
transforms = _module
transforms = _module
utils = _module
logger = _module
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


from torch.utils.data import Dataset


import torchvision.transforms.transforms as transformations


import numpy as np


import torch


from torchvision.datasets import *


from torchvision.datasets.folder import default_loader


from torch.utils.data import DataLoader


from torch.autograd import Variable


import random


from torch import nn


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import torch.nn as nn


from math import floor


import torch.distributions as distibutions


import torch.nn.functional as F


from torch.nn.utils import spectral_norm


from math import sqrt


from collections import namedtuple


from torch.autograd import grad


import torch.cuda as cuda


from torchvision import utils as vutils


from time import time


import matplotlib.pyplot as plt


import torch.distributions as distribution


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.onnx as onnx


from math import ceil


from torch.nn.init import *


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.conv import _single


from torch.nn.modules.conv import _pair


from torch.nn.modules.conv import _triple


from torch.nn.modules.batchnorm import _BatchNorm


from torch.optim import Adam


from torch.optim.lr_scheduler import StepLR


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, *inputs):
        return (input.half() for input in inputs)


class BatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, weight_init=None, bias_init=None):
        """

        :param num_features:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param weight_init:
        :param bias_init:
        """
        super(BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias_init is not None:
            bias_init(self.bias.data)


class BatchNorm2d(BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class Kaiming_Normal(object):

    def __init__(self, neg_slope=0, mode='fan_in', non_linearity='leaky_relu'):
        """

        :param neg_slope:
        :param mode:
        :param non_linearity:
        """
        self.neg_slope = neg_slope
        self.mode = mode
        self.non_linearity = non_linearity

    def __call__(self, tensor):
        return kaiming_normal_(tensor, self.neg_slope, self.mode, self.non_linearity)


class Constant(object):

    def __init__(self, value):
        """

        :param value:
        """
        self.value = value

    def __call__(self, tensor):
        return constant_(tensor, self.value)


class Zeros(Constant):

    def __init__(self):
        super(Zeros, self).__init__(0)


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class Embedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None, weight_init=None):
        """

        :param num_embeddings:
        :param embedding_dim:
        :param padding_idx:
        :param max_norm:
        :param norm_type:
        :param scale_grad_by_freq:
        :param sparse:
        :param _weight:
        :param weight_init:
        """
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        if weight_init is not None:
            weight_init(self.weight.data)


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_class, eps=1e-05, momentum=0.1, track_running_stats=True):
        """

        :param num_features:
        :param num_class:
        :param eps:
        :param momentum:
        :param track_running_stats:
        """
        super(ConditionalBatchNorm2d, self).__init__()
        self.bn = BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, track_running_stats=track_running_stats, affine=False)
        self.gamma_embed = Embedding(num_class, num_features)
        self.beta_embed = Embedding(num_class, num_features)
        self.gamma_embed.weight.data = torch.ones(self.gamma_embed.weight.size())
        self.beta_embed.weight.data = torch.zeros(self.gamma_embed.weight.size())

    def forward(self, input, class_id):
        input = input.float()
        class_id = class_id.long()
        out = self.bn(input)
        gamma = self.gamma_embed(class_id).squeeze(1).unsqueeze(2).unsqueeze(3)
        beta = self.beta_embed(class_id).squeeze(1).unsqueeze(2).unsqueeze(3)
        out = gamma * out.type(gamma.dtype) + beta
        return out


class Xavier_Uniform(object):

    def __init__(self, gain=1):
        """

        :param gain:
        """
        self.gain = gain

    def __call__(self, tensor):
        return xavier_uniform_(tensor, self.gain)


class GeneratorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes=0, upsample_size=1, kernel_size=3, activation=nn.ReLU(), conv_groups=1, dropout_ratio=0):
        """

        :param in_channels:
        :param out_channels:
        :param num_classes:
        :param upsample_size:
        :param kernel_size:
        :param activation:
        :param conv_groups:
        :param dropout_ratio:
        """
        super(GeneratorResBlock, self).__init__()
        padding = floor(kernel_size / 2)
        self.activation = activation
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.dropout = nn.Dropout(dropout_ratio)
        self.conv1 = spectral_norm(Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform(sqrt(2)), groups=conv_groups))
        self.conv2 = spectral_norm(Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform(sqrt(2)), groups=conv_groups))
        if num_classes > 0:
            self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
            self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        else:
            self.bn1 = BatchNorm2d(in_channels)
            self.bn2 = BatchNorm2d(out_channels)
        self.res_upsample = nn.Sequential()
        if in_channels != out_channels or upsample_size > 1:
            self.res_upsample = Conv2d(in_channels, out_channels, kernel_size=1, weight_init=Xavier_Uniform())

    def forward(self, inputs, labels=None):
        res = inputs
        if labels is not None:
            inputs = self.bn1(inputs, labels)
        else:
            inputs = self.bn1(inputs)
        inputs = self.dropout(self.conv1(self.activation(inputs)))
        if self.upsample_size > 1:
            inputs = F.interpolate(inputs, scale_factor=self.upsample_size)
        if labels is not None:
            inputs = self.bn2(inputs, labels)
        else:
            inputs = self.bn2(inputs)
        inputs = self.conv2(self.activation(inputs))
        if self.upsample_size > 1:
            return inputs + F.interpolate(self.res_upsample(res), scale_factor=self.upsample_size)
        else:
            return inputs + self.res_upsample(res)


class Xavier_Normal(object):

    def __init__(self, gain=1):
        """

        :param gain:
        """
        self.gain = gain

    def __call__(self, tensor):
        return xavier_normal_(tensor, self.gain)


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, weight_init=Xavier_Normal(), bias_init=Zeros()):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param weight_init:
        :param bias_init:
        """
        super(Linear, self).__init__(in_features, out_features, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class Normal(object):

    def __init__(self, mean=0, std=1):
        """

        :param mean:
        :param std:
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return normal_(tensor, self.mean, self.std)


class SelfAttention(nn.Module):

    def __init__(self, in_channels, weight_init=Kaiming_Normal(), bias_init=Zeros(), use_bias=False):
        """

        :param in_channels:
        :param weight_init:
        :param bias_init:
        :param use_bias:
        """
        super(SelfAttention, self).__init__()
        self.q = Conv2d(in_channels, in_channels // 8, kernel_size=1, weight_init=weight_init, bias_init=bias_init, bias=use_bias)
        self.k = Conv2d(in_channels, in_channels // 8, kernel_size=1, weight_init=weight_init, bias_init=bias_init, bias=use_bias)
        self.v = Conv2d(in_channels, in_channels, kernel_size=1, weight_init=weight_init, bias_init=bias_init, bias=use_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.atten_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, input):
        batch_size, channels, width, height = input.size()
        res = input
        queries = self.q(input).view(batch_size, -1, width * height).permute(0, 2, 1)
        keys = self.k(input).view(batch_size, -1, width * height)
        values = self.v(input).view(batch_size, -1, width * height)
        atten_ = self.softmax(torch.bmm(queries, keys)).permute(0, 2, 1)
        atten_values = torch.bmm(values, atten_).view(batch_size, channels, width, height)
        return self.atten_weight * atten_values + res


class ResGenerator(nn.Module):

    def __init__(self, output_size, num_classes=0, latent_size=100, kernel_size=3, activation=nn.ReLU(), conv_groups=1, attention=False, dropout_ratio=0):
        super(ResGenerator, self).__init__()
        padding = floor(kernel_size / 2)
        self.num_classes = num_classes
        output_channels = output_size[0]
        self.size = output_size[1]
        self.fc = Linear(latent_size, 4 * 4 * self.size * 8)
        current_size = 4
        self.layers = nn.ModuleList()
        in_channels = self.size * 8
        while current_size < self.size:
            self.layers.append(GeneratorResBlock(in_channels, in_channels // 2, num_classes, upsample_size=2, kernel_size=kernel_size, activation=activation, conv_groups=conv_groups if current_size == 4 else 1, dropout_ratio=dropout_ratio))
            current_size *= 2
            in_channels = in_channels // 2
            if current_size == self.size // 2 and attention:
                self.layers.append(SelfAttention(in_channels))
        self.net = nn.Sequential(BatchNorm2d(in_channels, weight_init=Normal(1.0, 0.02)), Conv2d(in_channels, output_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform()), nn.Tanh())

    def forward(self, inputs, labels=None):
        outputs = self.fc(inputs).view(-1, self.size * 8, 4, 4)
        for layer in self.layers:
            if self.num_classes > 1 and not isinstance(layer, SelfAttention):
                outputs = layer(outputs, labels)
            else:
                outputs = layer(outputs)
        return self.net(outputs)


class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class StandardGeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, num_classes=0, activation=nn.LeakyReLU(0.2), conv_groups=1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param stride:
        :param num_classes:
        :param activation:
        :param conv_groups:
        """
        super(StandardGeneratorBlock, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.conv = spectral_norm(ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, weight_init=Xavier_Uniform(), groups=conv_groups))
        if num_classes > 0:
            self.bn = ConditionalBatchNorm2d(out_channels, num_classes)
        else:
            self.bn = BatchNorm2d(out_channels)

    def forward(self, inputs, labels=None):
        inputs = self.conv(inputs)
        if labels is not None:
            inputs = self.bn(inputs, labels)
        else:
            inputs = self.bn(inputs)
        inputs = self.activation(inputs)
        return inputs


class StandardGenerator(nn.Module):

    def __init__(self, output_size, num_classes=0, latent_size=100, activation=nn.LeakyReLU(0.2), conv_groups=1, attention=False, dropout_ratio=0):
        super(StandardGenerator, self).__init__()
        output_channels = output_size[0]
        self.size = output_size[1]
        self.latent_size = latent_size
        self.num_classes = num_classes
        current_size = 4
        self.layers = nn.ModuleList()
        in_channels = self.size * 8
        self.layers.append(StandardGeneratorBlock(latent_size, in_channels, num_classes=num_classes, kernel_size=4, padding=0, stride=1, activation=activation))
        while current_size < self.size:
            current_size *= 2
            if current_size < self.size:
                self.layers.append(StandardGeneratorBlock(in_channels, in_channels // 2, num_classes=num_classes, kernel_size=4, stride=2, padding=1, activation=activation, conv_groups=conv_groups))
                self.layers.append(nn.Dropout(dropout_ratio))
                in_channels = in_channels // 2
                if current_size == self.size // 2 and attention:
                    self.layers.append(SelfAttention(in_channels))
        self.final_conv = spectral_norm(ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1, weight_init=Xavier_Uniform()))

    def forward(self, inputs, labels=None):
        outputs = inputs.view(-1, self.latent_size, 1, 1)
        for layer in self.layers:
            if self.num_classes > 1 and not isinstance(layer, nn.Dropout) and not isinstance(layer, SelfAttention):
                outputs = layer(outputs, labels)
            else:
                outputs = layer(outputs)
        return torch.tanh(self.final_conv(outputs))


class DiscriminatorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample_size=1, kernel_size=3, activation=nn.ReLU(), initial_activation=True, conv_groups=1, dropout_ratio=0):
        """

        :param in_channels:
        :param out_channels:
        :param downsample_size:
        :param kernel_size:
        :param activation:
        :param initial_activation:
        :param conv_groups:
        """
        super(DiscriminatorResBlock, self).__init__()
        padding = floor(kernel_size / 2)
        self.activation = activation
        self.initial_activation = initial_activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.conv1 = spectral_norm(Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform(), groups=conv_groups))
        self.conv2 = spectral_norm(Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, weight_init=Xavier_Uniform(), groups=conv_groups))
        self.downsample = nn.Sequential()
        if downsample_size > 1:
            self.downsample = nn.AvgPool2d(kernel_size=downsample_size)
        self.res_downsample = nn.Sequential()
        if in_channels != out_channels or downsample_size > 1:
            self.res_downsample = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size=1, weight_init=Xavier_Uniform(sqrt(2))), nn.AvgPool2d(kernel_size=downsample_size))

    def forward(self, inputs):
        res = inputs
        if self.initial_activation:
            inputs = self.activation(inputs)
        inputs = self.conv1(inputs)
        inputs = self.dropout(self.activation(inputs))
        inputs = self.conv2(inputs)
        inputs = self.downsample(inputs)
        return inputs + self.res_downsample(res)


class _GlobalPoolNd(nn.Module):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(_GlobalPoolNd, self).__init__()
        self.flatten = flatten

    def pool(self, input):
        """

        :param input:
        :return:
        """
        raise NotImplementedError()

    def forward(self, input):
        """

        :param input:
        :return:
        """
        input = self.pool(input)
        size_0 = input.size(1)
        return input.view(-1, size_0) if self.flatten else input


class GlobalAvgPool2d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalAvgPool2d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool2d(input, 1)


class ResProjectionDiscriminator(nn.Module):

    def __init__(self, input_size, num_classes=0, kernel_size=3, activation=nn.ReLU(), attention=True, apply_sigmoid=False, conv_groups=1, dropout_ratio=0):
        super(ResProjectionDiscriminator, self).__init__()
        self.num_classes = num_classes
        in_channels = input_size[0]
        out_channels = in_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid
        layers = [DiscriminatorResBlock(in_channels, size, kernel_size=kernel_size, activation=activation, initial_activation=False, dropout_ratio=dropout_ratio)]
        current_size = size
        in_channels = size
        while current_size > 4:
            layers.append(DiscriminatorResBlock(in_channels, in_channels * 2, kernel_size=kernel_size, downsample_size=2, activation=activation, conv_groups=conv_groups, dropout_ratio=dropout_ratio))
            current_size /= 2
            in_channels *= 2
            if current_size == size // 2 and attention:
                layers.append(SelfAttention(in_channels))
        layers.append(GlobalAvgPool2d())
        self.fc = spectral_norm(Linear(in_channels, out_channels, weight_init=Xavier_Uniform()))
        self.net = nn.Sequential(*layers)
        if self.num_classes > 1:
            self.embed = spectral_norm(Embedding(num_classes, in_channels, weight_init=Xavier_Uniform()))

    def forward(self, inputs, labels=None):
        outputs = self.net(inputs)
        linear_out = self.fc(outputs)
        if self.num_classes > 1:
            embed = self.embed(labels.long()).squeeze(1)
            size = outputs.size(1)
            dot = torch.bmm(outputs.view(-1, 1, size), embed.view(-1, size, 1)).squeeze(2)
            return torch.sigmoid(linear_out + dot) if self.apply_sigmoid else linear_out + dot
        else:
            return torch.sigmoid(linear_out) if self.apply_sigmoid else linear_out


class Flatten(nn.Module):

    def __init__(self, batch_first=True):
        """

        :param batch_first:
        """
        super(Flatten, self).__init__()
        self.batch_first = batch_first

    def forward(self, inputs):
        if self.batch_first:
            size = torch.prod(torch.LongTensor(list(inputs.size())[1:])).item()
            return inputs.view(-1, size)
        else:
            size = torch.prod(torch.LongTensor(list(inputs.size())[:len(inputs.size()) - 1])).item()
            return inputs.view(size, -1)


class StandardDiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, activation=nn.LeakyReLU(0.2), use_bn=False, conv_groups=1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param stride:
        :param activation:
        :param use_bn:
        :param conv_groups:
        """
        super(StandardDiscriminatorBlock, self).__init__()
        self.activation = activation
        self.conv = spectral_norm(Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, weight_init=Xavier_Uniform(), groups=conv_groups))
        self.bn = nn.Sequential()
        if use_bn:
            self.bn = BatchNorm2d(out_channels, weight_init=Normal(1.0, 0.02))

    def forward(self, inputs):
        return self.activation(self.bn(self.conv(inputs)))


class StandardProjectionDiscriminator(nn.Module):

    def __init__(self, input_size, num_classes=0, activation=nn.LeakyReLU(0.2), attention=True, apply_sigmoid=True, use_bn=False, conv_groups=1, dropout_ratio=0):
        super(StandardProjectionDiscriminator, self).__init__()
        self.num_classes = num_classes
        in_channels = input_size[0]
        out_channels = in_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid
        layers = [StandardDiscriminatorBlock(in_channels, size, kernel_size=3, stride=1, padding=1, use_bn=use_bn, activation=activation)]
        current_size = size
        in_channels = size
        while current_size > 4:
            layers.append(StandardDiscriminatorBlock(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1, use_bn=use_bn, conv_groups=conv_groups))
            layers.append(nn.Dropout(dropout_ratio))
            current_size /= 2
            in_channels *= 2
            if current_size == size // 2 and attention:
                layers.append(SelfAttention(in_channels))
        layers.append(Flatten())
        self.fc = spectral_norm(Linear(in_channels * 16, 1, weight_init=Xavier_Uniform()))
        self.net = nn.Sequential(*layers)
        if self.num_classes > 1:
            self.embed = spectral_norm(Embedding(num_classes, in_channels * 16, weight_init=Xavier_Uniform()))

    def forward(self, inputs, labels=None):
        outputs = self.net(inputs)
        linear_out = self.fc(outputs)
        if self.num_classes > 1:
            embed = self.embed(labels.long()).squeeze(1)
            size = outputs.size(1)
            dot = torch.bmm(outputs.view(-1, 1, size), embed.view(-1, size, 1)).squeeze(2)
            return torch.sigmoid(linear_out + dot) if self.apply_sigmoid else linear_out + dot
        else:
            return torch.sigmoid(linear_out) if self.apply_sigmoid else linear_out


class DCGANGenerator(nn.Module):

    def __init__(self, latent_size, output_size, dropout_ratio=0.0, use_bias=False, num_gpus=1):
        super(DCGANGenerator, self).__init__()
        assert output_size[1] >= 32
        self.num_gpus = num_gpus
        in_channels = latent_size[0]
        multiplier = 8
        out_size = output_size[1]
        layers = [ConvTranspose2d(in_channels=in_channels, out_channels=int(out_size * multiplier), kernel_size=4, stride=1, padding=0, bias=use_bias, weight_init=Normal(0.0, 0.02)), BatchNorm2d(int(out_size * multiplier), weight_init=Normal(1.0, 0.02)), nn.ReLU(inplace=True), nn.Dropout(dropout_ratio)]
        in_channels = int(out_size * multiplier)
        size = 4 * latent_size[1]
        while size < output_size[1]:
            multiplier /= 2
            size *= 2
            if size < int(out_size * multiplier):
                out_channels = int(out_size * multiplier)
            else:
                out_channels = out_size
            if size == output_size[1]:
                layers.append(ConvTranspose2d(in_channels=in_channels, out_channels=output_size[0], kernel_size=4, stride=2, padding=1, bias=use_bias, weight_init=Normal(0.0, 0.02)))
                layers.append(nn.Tanh())
            else:
                layers.append(ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias, weight_init=Normal(0.0, 0.02)))
                layers.append(BatchNorm2d(out_channels, weight_init=Normal(1.0, 0.02)))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_ratio))
                in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        if inputs.is_cuda and self.num_gpus > 1:
            out = nn.parallel.data_parallel(self.net, inputs, range(self.num_gpus))
        else:
            out = self.net(inputs)
        return out


class DCGANDiscriminator(nn.Module):

    def __init__(self, input_size, dropout_ratio=0.0, use_bias=False, num_gpus=1, apply_sigmoid=True):
        super(DCGANDiscriminator, self).__init__()
        assert input_size[1] >= 32
        self.num_gpus = num_gpus
        input_channels = input_size[0]
        in_channels = input_channels
        size = input_size[1]
        self.apply_sigmoid = apply_sigmoid
        channel_multiplier = 1
        out_channels = size
        layers = []
        while size > 4:
            layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias, weight_init=Normal(0.0, 0.02)))
            if size != input_size[1]:
                layers.append(BatchNorm2d(out_channels, weight_init=Normal(1.0, 0.02)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2
            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier
        layers.append(Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, padding=0, bias=use_bias, weight_init=Normal(0.0, 0.02)))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.num_gpus))
        else:
            output = self.net(input)
        return torch.sigmoid(output.view(-1, 1)) if self.apply_sigmoid else output.view(-1, 1)


class WGANDiscriminator(nn.Module):

    def __init__(self, input_size, dropout_ratio=0.0, use_bias=False, num_gpus=1):
        super(WGANDiscriminator, self).__init__()
        assert input_size[1] >= 32
        self.num_gpus = num_gpus
        input_channels = input_size[0]
        in_channels = input_channels
        size = input_size[1]
        channel_multiplier = 1
        out_channels = size
        layers = []
        while size > 4:
            layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias, weight_init=Normal(0.0, 0.02)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_ratio))
            if channel_multiplier < 8:
                channel_multiplier *= 2
            size /= 2
            in_channels = out_channels
            out_channels = input_size[1] * channel_multiplier
        layers.append(Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, padding=0, bias=use_bias, weight_init=Normal(0.0, 0.02)))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.num_gpus))
        else:
            output = self.net(input)
        return output.view(-1, 1).squeeze(1)


class MLPGenerator(nn.Module):

    def __init__(self, latent_size, output_size, hidden_dims=512, depth=4, dropout_ratio=0.0, num_gpus=1):
        """

        :param latent_size:
        :param output_size:
        :param hidden_dims:
        :param depth:
        :param dropout_ratio:
        :param num_gpus:
        """
        super(MLPGenerator, self).__init__()
        self.num_gpus = num_gpus
        self.output_size = output_size
        layers = []
        layers.append(Linear(latent_size, hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))
        for i in range(depth - 2):
            layers.append(Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))
        layers.append(Linear(hidden_dims, output_size[0] * output_size[1] * output_size[2]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.num_gpus))
        else:
            output = self.net(input)
        return output.view(-1, self.output_size[0], self.output_size[1], self.output_size[2])


class MLPDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_dims=512, depth=4, dropout_ratio=0.0, num_gpus=1, apply_sigmoid=True):
        """

        :param input_size:
        :param hidden_dims:
        :param depth:
        :param dropout_ratio:
        :param num_gpus:
        :param apply_sigmoid:
        """
        super(MLPDiscriminator, self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size
        self.apply_sigmoid = apply_sigmoid
        layers = []
        layers.append(Linear(input_size[0] * input_size[1] * input_size[2], hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))
        for i in range(depth - 2):
            layers.append(Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))
        layers.append(Linear(hidden_dims, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.num_gpus))
        else:
            output = self.net(input)
        return torch.sigmoid(output.view(-1, 1)) if self.apply_sigmoid else output.view(-1, 1)


class WMLPDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_dims=512, depth=4, dropout_ratio=0.0, num_gpus=1):
        """

        :param input_size:
        :param hidden_dims:
        :param depth:
        :param dropout_ratio:
        :param num_gpus:
        """
        super(WMLPDiscriminator, self).__init__()
        self.num_gpus = num_gpus
        self.input_size = input_size
        layers = []
        layers.append(Linear(input_size[0] * input_size[1] * input_size[2], hidden_dims))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout_ratio))
        for i in range(depth - 2):
            layers.append(Linear(hidden_dims, hidden_dims))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_ratio))
        layers.append(Linear(hidden_dims, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, input.size(1) * input.size(2) * input.size(3))
        if input.is_cuda and self.num_gpus > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.num_gpus))
        else:
            output = self.net(input)
        return output.view(-1, 1)


class MultiSequential(nn.Sequential):

    def __init__(self, *args):
        super(MultiSequential, self).__init__(*args)

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class Conv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConv1d(nn.Conv1d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConv1d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, padding, dilation, in_channels, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConv2d(nn.Conv2d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConv2d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, pa2ding, dilation, in_channels, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConv3d(nn.Conv3d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConv3d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, pa2ding, dilation, in_channels, bias)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class ConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConvTranspose1d(nn.ConvTranspose1d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConvTranspose1d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, padding, output_padding, in_channels, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConvTranspose2d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, padding, output_padding, in_channels, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class DepthwiseConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, multiplier=1, weight_init=Kaiming_Normal(), bias_init=Zeros()):
        super(DepthwiseConvTranspose3d, self).__init__(in_channels, in_channels * multiplier, kernel_size, stride, padding, output_padding, in_channels, bias, dilation)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias and bias_init is not None:
            bias_init(self.bias.data)


class Reshape(nn.Module):

    def __init__(self, output_shape, batch_first=True):
        """

        :param output_shape:
        :param batch_first:
        """
        super(Reshape, self).__init__()
        self.output_shape = output_shape
        self.batch_first = batch_first

    def forward(self, inputs):
        if isinstance(self.output_shape, int):
            size = [self.output_shape]
        else:
            size = list(self.output_shape)
        if self.batch_first:
            input_total_size = torch.prod(torch.LongTensor(list(inputs.size())[1:])).item()
        else:
            input_total_size = torch.prod(torch.LongTensor(list(inputs.size())[:len(inputs.size()) - 1])).item()
        target_total_size = torch.prod(torch.LongTensor(size)).item()
        if input_total_size != target_total_size:
            raise ValueError(' Reshape must preserve total dimension, input size: {} and output size: {}'.format(input.size()[1:], self.output_shape))
        size = list(size)
        if self.batch_first:
            size = tuple([-1] + size)
        else:
            size = tuple(size + [-1])
        outputs = inputs.view(size)
        return outputs


class GlobalAvgPool1d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalAvgPool1d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool1d(input, 1)


class GlobalAvgPool3d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalAvgPool3d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool3d(input, 1)


class GlobalMaxPool1d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalMaxPool1d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool1d(input, 1)


class GlobalMaxPool2d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalMaxPool2d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool2d(input, 1)


class GlobalMaxPool3d(_GlobalPoolNd):

    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalMaxPool3d, self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_max_pool3d(input, 1)


class RNNBase(nn.RNNBase):

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, weight_init=None):
        """

        :param mode:
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param bias:
        :param batch_first:
        :param dropout:
        :param bidirectional:
        :param weight_init:
        """
        super(RNNBase, self).__init__(mode, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        if weight_init is not None:
            for weight in super(RNNBase, self).parameters():
                weight_init(weight)


class RNN(RNNBase):

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'
        super(RNN, self).__init__(mode, *args, **kwargs)


class GRU(RNNBase):

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super(GRU, self).__init__('GRU', *args, **kwargs)


class LSTM(RNNBase):

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)


class GroupNorm(nn.GroupNorm):

    def __init__(self, *args, weight_init=None, bias_init=None):
        """

        :param args:
        :param weight_init:
        :param bias_init:
        """
        super(GroupNorm, self).__init__(*args)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias_init is not None:
            bias_init(self.bias.data)


class LayerNorm(nn.LayerNorm):

    def __init__(self, *args, weight_init=None, bias_init=None):
        """

        :param args:
        :param weight_init:
        :param bias_init:
        """
        super(LayerNorm, self).__init__(*args)
        if weight_init is not None:
            weight_init(self.weight.data)
        if bias_init is not None:
            bias_init(self.bias.data)


class BatchNorm1d(BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchNorm3d(BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConditionalBatchNorm2d,
     lambda: ([], {'num_features': 4, 'num_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (ConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvTranspose3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (DepthwiseConv1d,
     lambda: ([], {'in_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (DepthwiseConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (DepthwiseConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseConvTranspose3d,
     lambda: ([], {'in_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (DiscriminatorResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Embedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GeneratorResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalAvgPool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPDiscriminator,
     lambda: ([], {'input_size': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPGenerator,
     lambda: ([], {'latent_size': 4, 'output_size': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiSequential,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (RNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ResProjectionDiscriminator,
     lambda: ([], {'input_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reshape,
     lambda: ([], {'output_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'in_channels': 18}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
    (StandardDiscriminatorBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StandardGenerator,
     lambda: ([], {'output_size': [4, 4]}),
     lambda: ([torch.rand([4, 100, 1, 1])], {}),
     False),
    (StandardGeneratorBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (StandardProjectionDiscriminator,
     lambda: ([], {'input_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WMLPDiscriminator,
     lambda: ([], {'input_size': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
]

class Test_johnolafenwa_TorchFusion(_paritybench_base):
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

