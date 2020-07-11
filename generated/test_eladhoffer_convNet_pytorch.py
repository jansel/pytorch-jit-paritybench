import sys
_module = sys.modules[__name__]
del sys
autoaugment = _module
compare_experiments = _module
data = _module
evaluate = _module
main = _module
models = _module
alexnet = _module
densenet = _module
efficientnet = _module
evolved = _module
googlenet = _module
inception_resnet_v2 = _module
inception_v2 = _module
mnist = _module
mobilenet = _module
mobilenet_v2 = _module
modules = _module
activations = _module
batch_norm = _module
birelu = _module
bwn = _module
checkpoint = _module
evolved_modules = _module
fixed_proj = _module
fixup = _module
lp_norm = _module
quantize = _module
se = _module
resnet = _module
resnet_zi = _module
resnext = _module
vgg = _module
preprocess = _module
probe = _module
trainer = _module

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


import torchvision.datasets as datasets


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data import Subset


from torch._utils import _accumulate


from itertools import chain


from copy import deepcopy


import warnings


import time


import logging


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.distributed as dist


import torchvision.transforms as transforms


import torch.nn.functional as F


from collections import OrderedDict


import math


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn import BatchNorm1d as _BatchNorm1d


from torch.nn import BatchNorm2d as _BatchNorm2d


from torch.nn import BatchNorm3d as _BatchNorm3d


from torch.autograd.function import InplaceFunction


from torch.nn.parameter import Parameter


from torch.autograd import Function


from torch.utils.checkpoint import checkpoint


from torch.utils.checkpoint import checkpoint_sequential


from collections import namedtuple


from torch.autograd import Variable


from scipy.linalg import hadamard


import numpy as np


from torch.autograd.function import Function


from torchvision.models.vgg import vgg11


from torchvision.models.vgg import vgg11_bn


from torchvision.models.vgg import vgg13


from torchvision.models.vgg import vgg13_bn


from torchvision.models.vgg import vgg16


from torchvision.models.vgg import vgg16_bn


from torchvision.models.vgg import vgg19


from torchvision.models.vgg import vgg19_bn


import random


from torch.nn.utils import clip_grad_norm_


from random import sample


class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d(192), nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(384), nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d(256))
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(4096, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(4096, num_classes))
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.01, 'weight_decay': 0.0005, 'momentum': 0.9}, {'epoch': 10, 'lr': 0.005}, {'epoch': 15, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 20, 'lr': 0.0005}, {'epoch': 25, 'lr': 0.0001}]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_regime = [{'transform': transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])}]
        self.data_eval_regime = [{'transform': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])}]

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class CheckpointModule(nn.Module):

    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        if self.num_segments > 1:
            return checkpoint_sequential(self.module, self.num_segments, *inputs)
        else:
            return checkpoint(self.module, *inputs)


def _sum_tensor_scalar(tensor, scalar, expand_size):
    if scalar is not None:
        scalar = scalar.expand(expand_size).contiguous()
    else:
        return tensor
    if tensor is None:
        return scalar
    return tensor + scalar


class ZIConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, multiplier=False, pre_bias=True, post_bias=True):
        super(ZIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_channels)
        return nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, pre_bias=True, post_bias=True, multiplier=False):
    """3x3 convolution with padding"""
    return ZIConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias, pre_bias=pre_bias, post_bias=post_bias, multiplier=multiplier)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=1, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups, multiplier=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.residual_block is not None:
            residual = self.residual_block(residual)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ZIConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.conv3 = ZIConv2d(planes, planes * expansion, kernel_size=1, multiplier=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.residual_block is not None:
            residual = self.residual_block(residual)
        out += residual
        out = self.relu(out)
        return out


def init_model(model):

    def zi(layer, m, L):
        layer.weight.data.div_(math.pow(L, -1.0 / (2 * m - 2)))
    for m in model.modules():
        if isinstance(m, ZIConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
    for m in model.modules():
        if isinstance(m, Bottleneck):
            zi(m.conv1, 3, m.layer_depth)
            zi(m.conv2, 3, m.layer_depth)
            nn.init.constant_(m.conv3.weight, 0)
        elif isinstance(m, BasicBlock):
            zi(m.conv1, 2, m.layer_depth)
            nn.init.constant_(m.conv2.weight, 0)
    model.fc.weight.data.zero_()
    model.fc.bias.data.zero_()


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    @staticmethod
    def _create_features(num_features):
        return nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, checkpoint_segments=0):
        super(DenseNet, self).__init__()
        self.features = self._create_features(num_init_features)
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        if checkpoint_segments > 0:
            self.features = CheckpointModule(self.features, checkpoint_segments)
        self.classifier = nn.Linear(num_features, num_classes)
        init_model(self)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def weight_decay_config(value=0.0001, log=True):
    return {'name': 'WeightDecay', 'value': value, 'log': log, 'filter': {'parameter_name': lambda n: 'bias' not in n and 'multiplier' not in n}}


class DenseNet_imagenet(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, regime='normal', scale_lr=1, **kwargs):
        super(DenseNet_imagenet, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, **kwargs)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr), 'regularizer': weight_decay_config(0.0001)}, {'epoch': 5, 'lr': scale_lr * 0.1}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
        elif regime == 'small':
            scale_lr *= 4
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'lr': scale_lr * 0.1, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 256}, {'epoch': 80, 'input_size': 224, 'batch_size': 64}]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 1024}, {'epoch': 80, 'input_size': 224, 'batch_size': 512}]


class DenseNet_cifar(DenseNet):

    @staticmethod
    def _create_features(num_features):
        return nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False))]))

    def __init__(self, *kargs, **kwargs):
        super(DenseNet_cifar, self).__init__(*kargs, **kwargs)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 150, 'lr': 0.01}, {'epoch': 225, 'lr': 0.001}]

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


@torch.jit.script
def hard_sigmoid(x):
    return F.relu6(x + 3).div_(6)


@torch.jit.script
def hard_swish(x):
    return x * hard_sigmoid(x)


class HardSwish(nn.Module):

    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return hard_swish(x)


@torch.jit.script
def swish(x):
    return x * x.sigmoid()


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)


class ConvBNAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, *kargs, **kwargs):
        hard_act = kwargs.pop('hard_act', False)
        kwargs.setdefault('bias', False)
        super(ConvBNAct, self).__init__(nn.Conv2d(in_channels, out_channels, *kargs, **kwargs), nn.BatchNorm2d(out_channels), HardSwish() if hard_act else Swish())


class HardSigmoid(nn.Module):

    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return hard_sigmoid(x)


class SESwishBlock(nn.Module):
    """ squeeze-excite block for MBConv """

    def __init__(self, in_channels, out_channels=None, interm_channels=None, ratio=None, hard_act=False):
        super(SESwishBlock, self).__init__()
        assert not (interm_channels is None and ratio is None)
        interm_channels = interm_channels or in_channels // ratio
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.activation = HardSwish() if hard_act else Swish(),
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(nn.Linear(in_channels, interm_channels), HardSwish() if hard_act else Swish(), nn.Linear(interm_channels, out_channels), HardSigmoid() if hard_act else nn.Sigmoid())

    def forward(self, x):
        x_avg = self.global_pool(x).flatten(1, -1)
        mask = self.transform(x_avg)
        return x * mask.unsqueeze(-1).unsqueeze(-1)


def drop_connect(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob).float()
        mask.div_(keep_prob)
        x = x.mul(mask)
    return x


class MBConv(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=1, kernel_size=3, stride=1, padding=1, se_ratio=0.25, hard_act=False):
        expanded = in_channels * expansion
        super(MBConv, self).__init__()
        self.add_res = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(ConvBNAct(in_channels, expanded, 1, hard_act=hard_act) if expanded != in_channels else nn.Identity(), ConvBNAct(expanded, expanded, kernel_size, stride=stride, padding=padding, groups=expanded, hard_act=hard_act), SESwishBlock(expanded, expanded, int(in_channels * se_ratio), hard_act=hard_act) if se_ratio > 0 else nn.Identity(), nn.Conv2d(expanded, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.drop_prob = 0

    def forward(self, x):
        out = self.block(x)
        if self.add_res:
            if self.training and self.drop_prob > 0.0:
                x = drop_connect(x, self.drop_prob)
            out += x
        return out


class MBConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, num, expansion=1, kernel_size=3, stride=1, padding=1, se_ratio=0.25, hard_act=False):
        kwargs = dict(expansion=expansion, kernel_size=kernel_size, stride=stride, padding=padding, se_ratio=se_ratio, hard_act=hard_act)
        first_conv = MBConv(in_channels, out_channels, **kwargs)
        kwargs['stride'] = 1
        super(MBConvBlock, self).__init__(first_conv, *[MBConv(out_channels, out_channels, **kwargs) for _ in range(num - 1)])


def modify_drop_connect_rate(model, value, log=True):
    for m in model.modules():
        if hasattr(m, 'drop_prob'):
            if log and m.drop_prob != value:
                logging.debug('Modified drop-path rate from %s to %s' % (m.drop_prob, value))
            m.drop_prob = value


class EfficientNet(nn.Module):

    def __init__(self, width_coeff=1, depth_coeff=1, resolution=224, se_ratio=0.25, regime='cosine', num_classes=1000, scale_lr=1, dropout_rate=0.2, drop_connect_rate=0.2, num_epochs=200, hard_act=False):
        super(EfficientNet, self).__init__()

        def channels(base_channels, coeff=width_coeff, divisor=8, min_channels=None):
            if coeff == 1:
                return base_channels
            min_channels = min_channels or divisor
            channels = base_channels * coeff
            channels = max(min_channels, int(base_channels + divisor / 2) // divisor * divisor)
            if channels < 0.9 * base_channels:
                channels += divisor
            return int(channels)

        def repeats(repeats, coeff=depth_coeff):
            return int(math.ceil(coeff * repeats))

        def config(out_channels, num, expansion=1, kernel_size=3, stride=1, padding=None, se_ratio=se_ratio, hard_act=hard_act):
            padding = padding or int((kernel_size - 1) // 2)
            return {'out_channels': channels(out_channels), 'num': repeats(num), 'expansion': expansion, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'se_ratio': se_ratio, 'hard_act': hard_act}
        stages = [config(16, num=1, expansion=1, kernel_size=3, stride=1), config(24, num=2, expansion=6, kernel_size=3, stride=2), config(40, num=2, expansion=6, kernel_size=5, stride=2), config(80, num=3, expansion=6, kernel_size=3, stride=2), config(112, num=3, expansion=6, kernel_size=5, stride=1), config(192, num=4, expansion=6, kernel_size=5, stride=2), config(320, num=1, expansion=6, kernel_size=3, stride=1)]
        layers = []
        for i in range(len(stages)):
            in_channel = channels(32) if i == 0 else stages[i - 1]['out_channels']
            layers.append(MBConvBlock(in_channel, **stages[i]))
        self.features = nn.Sequential(ConvBNAct(3, channels(32), 3, 2, 1, hard_act=hard_act), *layers, ConvBNAct(channels(320), channels(1280), 1), nn.AdaptiveAvgPool2d(1), nn.Dropout(dropout_rate, True))
        self.classifier = nn.Linear(channels(1280), num_classes)
        init_model(self)

        def increase_drop_connect(epoch):
            return lambda : modify_drop_connect_rate(self, min(drop_connect_rate, drop_connect_rate * epoch / float(num_epochs)))
        if regime == 'paper':

            def config_by_epoch(epoch):
                return {'lr': scale_lr * 0.016 * 0.97 ** round(epoch / 2.4), 'execute': increase_drop_connect(epoch)}
            """RMSProp optimizer with
            decay 0.9 and momentum 0.9;
            weight decay 1e-5; initial learning rate 0.256 that decays
            by 0.97 every 2.4 epochs"""
            self.regime = [{'optimizer': 'RMSprop', 'alpha': 0.9, 'momentum': 0.9, 'lr': scale_lr * 0.016, 'regularizer': weight_decay_config(1e-05), 'epoch_lambda': config_by_epoch}]
        elif regime == 'cosine':

            def cosine_anneal_lr(epoch, base_lr=0.025, T_max=num_epochs, eta_min=0.0001):
                return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

            def config_by_epoch(epoch):
                return {'lr': cosine_anneal_lr(epoch, base_lr=scale_lr * 0.1, T_max=num_epochs), 'execute': increase_drop_connect(epoch)}
            self.regime = [{'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-05), 'epoch_lambda': config_by_epoch}]
        self.data_regime = [{'input_size': resolution, 'autoaugment': True}]
        self.data_eval_regime = [{'input_size': resolution, 'scale_size': int(resolution * 8 / 7)}]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.flatten(1, -1))
        return x


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, channels, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(channels, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, channels, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(channels, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


OPS = {'avg_pool_3x3': lambda channels, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), 'max_pool_3x3': lambda channels, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda channels, stride, affine: Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine), 'sep_conv_3x3': lambda channels, stride, affine: SepConv(channels, channels, 3, stride, 1, affine=affine), 'sep_conv_5x5': lambda channels, stride, affine: SepConv(channels, channels, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda channels, stride, affine: SepConv(channels, channels, 7, stride, 3, affine=affine), 'dil_conv_3x3': lambda channels, stride, affine: DilConv(channels, channels, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda channels, stride, affine: DilConv(channels, channels, 5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda channels, stride, affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(channels, channels, (1, 7), stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(channels, channels, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(channels, affine=affine))}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


GENOTYPES = dict(NASNet=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=[4, 5, 6]), AmoebaNet=Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=[4, 5, 6], reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 2), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)], reduce_concat=[3, 4, 6]), DARTS_V1=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5]), DARTS=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5]))


class DARTSCell(Cell):

    def __init__(self, *kargs, **kwargs):
        super(DARTSCell, self).__init__(GENOTYPES['DARTS'], *kargs, **kwargs)


def cosine_anneal_lr(epoch, base_lr=0.025, T_max=600.0, eta_min=0.0):
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2


def modify_drop_path_rate(model, value, log=True):
    if log and model.drop_path != value:
        logging.debug('Modified drop-path rate from %s to %s' % (model.drop_path, value))
    model.drop_path = value


class EvolvedNetworkCIFAR(nn.Module):

    def __init__(self, init_channels=36, num_classes=10, layers=20, auxiliary=True, aux_weight=0.4, drop_path=0.2, num_epochs=600, init_lr=0.025, cell_fn=DARTSCell):
        super(EvolvedNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path = drop_path
        stem_multiplier = 3
        channels = stem_multiplier * init_channels
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        prev2_channels, prev_channels, channels = channels, channels, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                channels *= 2
                reduction = True
            else:
                reduction = False
            cell = cell_fn(prev2_channels, prev_channels, channels, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            prev2_channels, prev_channels = prev_channels, cell.multiplier * channels
            if i == 2 * layers // 3:
                aux_channels = prev_channels
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(aux_channels, num_classes)

            def loss_fn(*kargs, **kwargs):
                return MultiOutputLoss([1.0, aux_weight], *kargs, **kwargs)
            self.criterion = loss_fn
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_channels, num_classes)

        def config_by_epoch(epoch):
            return {'lr': cosine_anneal_lr(epoch, base_lr=init_lr, T_max=float(num_epochs)), 'execute': lambda : modify_drop_path_rate(self, drop_path * epoch / float(num_epochs))}
        self.regime = [{'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(0.0003), 'epoch_lambda': config_by_epoch}]

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary:
            return logits, logits_aux
        else:
            return logits


class EvolvedNetworkImageNet(nn.Module):

    def __init__(self, init_channels=36, num_classes=1000, layers=20, auxiliary=True, aux_weight=0.4, drop_path=0.2, cell_fn=DARTSCell):
        super(EvolvedNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path = drop_path
        self.stem0 = nn.Sequential(nn.Conv2d(3, init_channels // 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(init_channels // 2), nn.ReLU(inplace=True), nn.Conv2d(init_channels // 2, init_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(init_channels))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(init_channels, init_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(init_channels))
        prev2_channels, prev_channels, channels = init_channels, init_channels, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                channels *= 2
                reduction = True
            else:
                reduction = False
            cell = cell_fn(prev2_channels, prev_channels, channels, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            prev2_channels, prev_channels = prev_channels, cell.multiplier * channels
            if i == 2 * layers // 3:
                aux_channels = prev_channels
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(aux_channels, num_classes)

            def loss_fn(*kargs, **kwargs):
                return MultiOutputLoss([1.0, aux_weight], *kargs, **kwargs)
            self.criterion = loss_fn
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary:
            return logits, logits_aux
        else:
            return logits


def conv_bn(in_planes, out_planes, kernel_size, stride=1, padding=0):
    """convolution with batchnorm, relu"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU())


class InceptionModule(nn.Module):

    def __init__(self, in_channels, n1x1_channels, n3x3r_channels, n3x3_channels, dn3x3r_channels, dn3x3_channels, pool_proj_channels=None, type_pool='avg', stride=1):
        super(InceptionModule, self).__init__()
        self.in_channels = in_channels
        self.n1x1_channels = n1x1_channels or 0
        pool_proj_channels = pool_proj_channels or 0
        self.stride = stride
        if n1x1_channels > 0:
            self.conv_1x1 = conv_bn(in_channels, n1x1_channels, 1, stride)
        else:
            self.conv_1x1 = None
        self.conv_3x3 = nn.Sequential(conv_bn(in_channels, n3x3r_channels, 1), conv_bn(n3x3r_channels, n3x3_channels, 3, stride, padding=1))
        self.conv_d3x3 = nn.Sequential(conv_bn(in_channels, dn3x3r_channels, 1), conv_bn(dn3x3r_channels, dn3x3_channels, 3, padding=1), conv_bn(dn3x3_channels, dn3x3_channels, 3, stride, padding=1))
        if type_pool == 'avg':
            self.pool = nn.AvgPool2d(3, stride, padding=1)
        elif type_pool == 'max':
            self.pool = nn.MaxPool2d(3, stride, padding=1)
        if pool_proj_channels > 0:
            self.pool = nn.Sequential(self.pool, conv_bn(in_channels, pool_proj_channels, 1))

    def forward(self, inputs):
        layer_outputs = []
        if self.conv_1x1 is not None:
            layer_outputs.append(self.conv_1x1(inputs))
        layer_outputs.append(self.conv_3x3(inputs))
        layer_outputs.append(self.conv_d3x3(inputs))
        layer_outputs.append(self.pool(inputs))
        output = torch.cat(layer_outputs, 1)
        return output


class Inception_v1_GoogLeNet(nn.Module):
    input_side = 227
    rescale = 255.0
    rgb_mean = [122.7717, 115.9465, 102.9801]
    rgb_std = [1, 1, 1]

    def __init__(self, num_classes=1000):
        super(Inception_v1_GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Sequential(OrderedDict([('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), bias=False)), ('7x7_s2_bn', nn.BatchNorm2d(64, affine=True)), ('relu1', nn.ReLU(True)), ('pool1', nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)))]))), ('conv2', nn.Sequential(OrderedDict([('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0), bias=False)), ('3x3_reduce_bn', nn.BatchNorm2d(64, affine=True)), ('relu1', nn.ReLU(True)), ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1), bias=False)), ('3x3_bn', nn.BatchNorm2d(192, affine=True)), ('relu2', nn.ReLU(True)), ('pool2', nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)))]))), ('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)), ('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)), ('pool3', nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))), ('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)), ('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)), ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)), ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)), ('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)), ('pool4', nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))), ('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)), ('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)), ('pool5', nn.AvgPool2d((7, 7), (1, 1))), ('drop5', nn.Dropout(0.2))]))
        self.classifier = nn.Linear(1024, self.num_classes)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.001, 'optimizer': 'Adam'}]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Concat(nn.Sequential):

    def __init__(self, *kargs, **kwargs):
        super(Concat, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        return torch.cat([m(inputs) for m in self._modules.values()], 1)


class block(nn.Module):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block, self).__init__()
        self.scale = scale
        self.activation = activation or (lambda x: x)

    def forward(self, inputs):
        branch0 = self.Branch_0(inputs)
        branch1 = self.Branch_1(inputs)
        if hasattr(self, 'Branch_2'):
            branch2 = self.Branch_2(inputs)
            tower_mixed = torch.cat([branch0, branch1, branch2], 1)
        else:
            tower_mixed = torch.cat([branch0, branch1], 1)
        tower_out = self.Conv2d_1x1(tower_mixed)
        output = self.activation(self.scale * tower_out + inputs)
        return output


class block35(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block35, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 32, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)), ('Conv2d_0b_3x3', conv_bn(32, 32, 3, padding=1))]))
        self.Branch_2 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)), ('Conv2d_0b_3x3', conv_bn(32, 48, 3, padding=1)), ('Conv2d_0c_3x3', conv_bn(48, 64, 3, padding=1))]))
        self.Conv2d_1x1 = conv_bn(128, in_planes, 1)


class block17(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block17, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 192, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 128, 1)), ('Conv2d_0b_1x7', conv_bn(128, 160, (1, 7), padding=(0, 3))), ('Conv2d_0c_7x1', conv_bn(160, 192, (7, 1), padding=(3, 0)))]))
        self.Conv2d_1x1 = conv_bn(384, in_planes, 1)


class block8(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block8, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 192, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 192, 1)), ('Conv2d_0b_1x7', conv_bn(192, 224, (1, 3), padding=(0, 1))), ('Conv2d_0c_7x1', conv_bn(224, 256, (3, 1), padding=(1, 0)))]))
        self.Conv2d_1x1 = conv_bn(448, in_planes, 1)


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(InceptionResnetV2, self).__init__()
        self.end_points = {}
        self.num_classes = num_classes
        self.stem = nn.Sequential(OrderedDict([('Conv2d_1a_3x3', conv_bn(3, 32, 3, stride=2, padding=1)), ('Conv2d_2a_3x3', conv_bn(32, 32, 3, padding=1)), ('Conv2d_2b_3x3', conv_bn(32, 64, 3)), ('MaxPool_3a_3x3', nn.MaxPool2d(3, 2)), ('Conv2d_3b_1x1', conv_bn(64, 80, 1)), ('Conv2d_4a_3x3', conv_bn(80, 192, 3)), ('MaxPool_5a_3x3', nn.MaxPool2d(3, 2))]))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_5b_b0_1x1', conv_bn(192, 96, 1))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_5b_b1_0a_1x1', conv_bn(192, 48, 1)), ('Conv2d_5b_b1_0b_5x5', conv_bn(48, 64, 5, padding=2))]))
        tower_conv2 = nn.Sequential(OrderedDict([('Conv2d_5b_b2_0a_1x1', conv_bn(192, 64, 1)), ('Conv2d_5b_b2_0b_3x3', conv_bn(64, 96, 3, padding=1)), ('Conv2d_5b_b2_0c_3x3', conv_bn(96, 96, 3, padding=1))]))
        tower_pool3 = nn.Sequential(OrderedDict([('AvgPool_5b_b3_0a_3x3', nn.AvgPool2d(3, stride=1, padding=1)), ('Conv2d_5b_b3_0b_1x1', conv_bn(192, 64, 1))]))
        self.mixed_5b = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_conv2), ('Branch_3', tower_pool3)]))
        self.blocks35 = nn.Sequential()
        for i in range(10):
            self.blocks35.add_module('Block35.%s' % i, block35(320, scale=0.17))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_6a_b0_0a_3x3', conv_bn(320, 384, 3, stride=2))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_6a_b1_0a_1x1', conv_bn(320, 256, 1)), ('Conv2d_6a_b1_0b_3x3', conv_bn(256, 256, 3, padding=1)), ('Conv2d_6a_b1_0c_3x3', conv_bn(256, 384, 3, stride=2))]))
        tower_pool = nn.Sequential(OrderedDict([('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))]))
        self.mixed_6a = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_pool)]))
        self.blocks17 = nn.Sequential()
        for i in range(20):
            self.blocks17.add_module('Block17.%s' % i, block17(1088, scale=0.1))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_1a_3x3', conv_bn(256, 384, 3, stride=2))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_1a_3x3', conv_bn(256, 64, 3, stride=2))]))
        tower_conv2 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_0b_3x3', conv_bn(256, 288, 3, padding=1)), ('Conv2d_1a_3x3', conv_bn(288, 320, 3, stride=2))]))
        tower_pool3 = nn.Sequential(OrderedDict([('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))]))
        self.mixed_7a = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_conv2), ('Branch_3', tower_pool3)]))
        self.blocks8 = nn.Sequential()
        for i in range(9):
            self.blocks8.add_module('Block8.%s' % i, block8(1856, scale=0.2))
        self.blocks8.add_module('Block8.9', block8(1856, scale=0.2, activation=None))
        self.conv_pool = nn.Sequential(OrderedDict([('Conv2d_7b_1x1', conv_bn(1856, 1536, 1)), ('AvgPool_1a_8x8', nn.AvgPool2d(8, 1)), ('Dropout', nn.Dropout(0.2))]))
        self.classifier = nn.Linear(1536, num_classes)
        self.aux_classifier = nn.Sequential(OrderedDict([('Conv2d_1a_3x3', nn.AvgPool2d(5, 3)), ('Conv2d_1b_1x1', conv_bn(1088, 128, 1)), ('Conv2d_2a_5x5', conv_bn(128, 768, 5)), ('Dropout', nn.Dropout(0.2)), ('Logits', conv_bn(768, num_classes, 1))]))


        class aux_loss(nn.Module):

            def __init__(self):
                super(aux_loss, self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) + 0.4 * self.loss(outputs[1], target)
        self.criterion = aux_loss
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]

    def forward(self, x):
        x = self.stem(x)
        x = self.mixed_5b(x)
        x = self.blocks35(x)
        x = self.mixed_6a(x)
        branch1 = self.blocks17(x)
        x = self.mixed_7a(branch1)
        x = self.blocks8(x)
        x = self.conv_pool(x)
        x = x.view(-1, 1536)
        output = self.classifier(x)
        if hasattr(self, 'aux_classifier'):
            branch1 = self.aux_classifier(branch1).view(-1, self.num_classes)
            output = [output, branch1]
        return output


def inception_v2(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return Inception_v2(num_classes=num_classes)


class Inception_v2(nn.Module):

    def __init__(self, num_classes=1000, aux_classifiers=True):
        super(inception_v2, self).__init__()
        self.num_classes = num_classes
        self.part1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.MaxPool2d(3, 2), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 192, 3, 1, 1, bias=False), nn.MaxPool2d(3, 2), nn.BatchNorm2d(192), nn.ReLU(), InceptionModule(192, 64, 64, 64, 64, 96, 32, 'avg'), InceptionModule(256, 64, 64, 96, 64, 96, 64, 'avg'), InceptionModule(320, 0, 128, 160, 64, 96, 0, 'max', 2))
        self.part2 = nn.Sequential(InceptionModule(576, 224, 64, 96, 96, 128, 128, 'avg'), InceptionModule(576, 192, 96, 128, 96, 128, 128, 'avg'), InceptionModule(576, 160, 128, 160, 128, 160, 96, 'avg'))
        self.part3 = nn.Sequential(InceptionModule(576, 96, 128, 192, 160, 192, 96, 'avg'), InceptionModule(576, 0, 128, 192, 192, 256, 0, 'max', 2), InceptionModule(1024, 352, 192, 320, 160, 224, 128, 'avg'), InceptionModule(1024, 352, 192, 320, 192, 224, 128, 'max'))
        self.main_classifier = nn.Sequential(nn.AvgPool2d(7, 1), nn.Dropout(0.2), nn.Conv2d(1024, self.num_classes, 1))
        if aux_classifiers:
            self.aux_classifier1 = nn.Sequential(nn.AvgPool2d(5, 3), conv_bn(576, 128, 1), conv_bn(128, 768, 4), nn.Dropout(0.2), nn.Conv2d(768, self.num_classes, 1))
            self.aux_classifier2 = nn.Sequential(nn.AvgPool2d(5, 3), conv_bn(576, 128, 1), conv_bn(128, 768, 4), nn.Dropout(0.2), nn.Conv2d(768, self.num_classes, 1))
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]


        class aux_loss(nn.Module):

            def __init__(self):
                super(aux_loss, self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) + 0.4 * (self.loss(outputs[1], target) + self.loss(outputs[2], target))
        self.criterion = aux_loss

    def forward(self, inputs):
        branch1 = self.part1(inputs)
        branch2 = self.part2(branch1)
        branch3 = self.part3(branch1)
        output = self.main_classifier(branch3).view(-1, self.num_classes)
        if hasattr(self, 'aux_classifier1'):
            branch1 = self.aux_classifier1(branch1).view(-1, self.num_classes)
            branch2 = self.aux_classifier2(branch2).view(-1, self.num_classes)
            output = [output, branch1, branch2]
        return output


class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.BatchNorm2d(32), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(128))
        self.classifier = nn.Conv2d(128, 10, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 10)
        return out


class DepthwiseSeparableFusedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        self.components = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.components(x)


def nearby_int(n):
    return int(round(n))


class MobileNet(nn.Module):

    def __init__(self, width=1.0, shallow=False, regime=None, num_classes=1000):
        super(MobileNet, self).__init__()
        num_classes = num_classes or 1000
        width = width or 1.0
        layers = [nn.Conv2d(3, nearby_int(width * 32), kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(nearby_int(width * 32)), nn.ReLU(inplace=True), DepthwiseSeparableFusedConv2d(nearby_int(width * 32), nearby_int(width * 64), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 64), nearby_int(width * 128), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 128), nearby_int(width * 128), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 128), nearby_int(width * 256), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 256), nearby_int(width * 256), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 256), nearby_int(width * 512), kernel_size=3, stride=2, padding=1)]
        if not shallow:
            layers += [DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1)]
        layers += [DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 1024), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 1024), nearby_int(width * 1024), kernel_size=3, stride=1, padding=1)]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nearby_int(width * 1024), num_classes)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_regime = [{'transform': transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])}]
        if regime == 'small':
            scale_lr = 4
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'lr': scale_lr * 0.1, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 512}, {'epoch': 80, 'input_size': 224, 'batch_size': 128}]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 1024}, {'epoch': 80, 'input_size': 224, 'batch_size': 512}]
        else:
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001}, {'epoch': 80, 'lr': 0.0001}]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ExpandedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=1, kernel_size=3, stride=1, padding=1, residual_block=None):
        expanded = in_channels * expansion
        super(ExpandedConv2d, self).__init__()
        self.add_res = stride == 1 and in_channels == out_channels
        self.residual_block = residual_block
        if expanded == in_channels:
            block = []
        else:
            block = [nn.Conv2d(in_channels, expanded, 1, bias=False), nn.BatchNorm2d(expanded), nn.ReLU6(inplace=True)]
        block += [nn.Conv2d(expanded, expanded, kernel_size, stride=stride, padding=padding, groups=expanded, bias=False), nn.BatchNorm2d(expanded), nn.ReLU6(inplace=True), nn.Conv2d(expanded, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        if self.add_res:
            if self.residual_block is not None:
                x = self.residual_block(x)
            out += x
        return out


def conv(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class MobileNet_v2(nn.Module):

    def __init__(self, width=1.0, regime=None, num_classes=1000, scale_lr=1):
        super(MobileNet_v2, self).__init__()
        in_channels = nearby_int(width * 32)
        layers_config = [dict(expansion=1, stride=1, out_channels=nearby_int(width * 16)), dict(expansion=6, stride=2, out_channels=nearby_int(width * 24)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 24)), dict(expansion=6, stride=2, out_channels=nearby_int(width * 32)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 32)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 32)), dict(expansion=6, stride=2, out_channels=nearby_int(width * 64)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 64)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 96)), dict(expansion=6, stride=2, out_channels=nearby_int(width * 160)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 160)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 160)), dict(expansion=6, stride=1, out_channels=nearby_int(width * 320))]
        self.features = nn.Sequential()
        self.features.add_module('conv0', conv(3, in_channels, kernel=3, stride=2, padding=1))
        for i, layer in enumerate(layers_config):
            layer['in_channels'] = in_channels
            in_channels = layer['out_channels']
            self.features.add_module('bottleneck' + str(i), ExpandedConv2d(**layer))
        out_channels = nearby_int(width * 1280)
        self.features.add_module('conv1', conv(in_channels, out_channels, kernel=1, stride=1, padding=0))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2, True), nn.Linear(out_channels, num_classes))
        init_model(self)
        if regime == 'small':
            scale_lr *= 4
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 512}, {'epoch': 80, 'input_size': 224, 'batch_size': 128}]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 128, 'scale_size': 160, 'batch_size': 1024}, {'epoch': 80, 'input_size': 224, 'batch_size': 512}]
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'lr': scale_lr * 0.1, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def has_parameters(m):
    return getattr(m, 'weight', None) is not None or getattr(m, 'bias', None) is not None


def has_running_stats(m):
    return getattr(m, 'running_mean', None) is not None or getattr(m, 'running_var', None) is not None


class BatchNorm1d(_BatchNorm1d):

    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm1d, self).forward(inputs)


class BatchNorm2d(_BatchNorm2d):

    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm2d, self).forward(inputs)


class BatchNorm3d(_BatchNorm3d):

    def forward(self, inputs):
        if not (has_parameters(self) or has_running_stats(self)):
            return inputs
        return super(BatchNorm3d, self).forward(inputs)


class MeanBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with mean-only normalization"""

    def __init__(self, num_features, momentum=0.1, bias=True):
        nn.Module.__init__(self)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.num_features = num_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if not (has_parameters(self) or has_running_stats(self)):
            return x
        if self.training:
            numel = x.size(0) * x.size(2) * x.size(3)
            mean = x.sum((0, 2, 3)) / numel
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(1 - self.momentum, mean)
        else:
            mean = self.running_mean
        if self.bias is not None:
            mean = mean - self.bias
        return x - mean.view(1, -1, 1, 1)

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, bias={has_bias}'.format(has_bias=self.bias is not None, **self.__dict__)


class BiReLUFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, inplace=False):
        if input.size(1) % 2 != 0:
            raise RuntimeError('dimension 1 of input must be multiple of 2, but got {}'.format(input.size(1)))
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        pos, neg = output.chunk(2, dim=1)
        pos.clamp_(min=0)
        neg.clamp_(max=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        grad_input = grad_output.masked_fill(output.eq(0), 0)
        return grad_input, None


def birelu(x, inplace=False):
    return BiReLUFunction().apply(x, inplace)


class BiReLU(nn.Module):
    """docstring for BiReLU."""

    def __init__(self, inplace=False):
        super(BiReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs):
        return birelu(inputs, inplace=self.inplace)


class NasNetCell(Cell):

    def __init__(self, *kargs, **kwargs):
        super(NasNetCell, self).__init__(GENOTYPES['NASNet'], *kargs, **kwargs)


class AmoebaNetCell(Cell):

    def __init__(self, *kargs, **kwargs):
        super(AmoebaNetCell, self).__init__(GENOTYPES['AmoebaNet'], *kargs, **kwargs)


class HadamardProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(sz))
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat)
        init_scale = 1.0 / math.sqrt(self.output_size)
        if fixed_scale is not None:
            self.scale = Variable(torch.Tensor([fixed_scale]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.Tensor([init_scale]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).uniform_(-init_scale, init_scale))
        else:
            self.register_parameter('bias', None)
        self.eps = 1e-08

    def forward(self, x):
        if not isinstance(self.scale, nn.Parameter):
            self.scale = self.scale.type_as(x)
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)
        out = -self.scale * nn.functional.linear(x, w[:self.output_size, :self.input_size])
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Proj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):
        super(Proj, self).__init__()
        if init_scale is not None:
            self.weight = nn.Parameter(torch.Tensor(1).fill_(init_scale))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).fill_(0))
        self.proj = Variable(torch.Tensor(output_size, input_size), requires_grad=False)
        torch.manual_seed(123)
        nn.init.orthogonal(self.proj)

    def forward(self, x):
        w = self.proj.type_as(x)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w)
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out


class LinearFixed(nn.Linear):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):
        super(LinearFixed, self).__init__(input_size, output_size, bias)
        self.scale = nn.Parameter(torch.Tensor(1).fill_(init_scale))

    def forward(self, x):
        w = self.weight / self.weight.norm(2, -1, keepdim=True)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w, self.bias)
        return out


class ZILinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, multiplier=False, pre_bias=True, post_bias=True):
        super(ZILinear, self).__init__(in_features, out_features, bias)
        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.0]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_features)
        return nn.functional.linear(x, weight, bias)


def _std(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.std()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).std(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).std(dim=0).view(*output_size)
    else:
        return _std(p.transpose(0, dim), 0).transpose(0, dim)


class LpBatchNorm2d(nn.Module):

    def __init__(self, num_features, dim=1, p=2, momentum=0.1, bias=True, eps=1e-05, noise=False):
        super(LpBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.p = p
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        p = self.p
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            Var = (torch.abs(t.transpose(1, 0) - mean) ** p).mean(0)
            scale = (Var + self.eps) ** (-1 / p)
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)
        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class TopkBatchNorm2d(nn.Module):

    def __init__(self, num_features, k=10, dim=1, momentum=0.1, bias=True, eps=1e-05, noise=False):
        super(TopkBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.k = k
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)
            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / (2 * np.log(A.size(0))) ** 0.5
            MeanTOPK = torch.topk(A, self.k, dim=0)[0].mean(0) * const
            scale = 1 / (MeanTOPK + self.eps)
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)
        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class GhostTopkBatchNorm2d(nn.Module):

    def __init__(self, num_features, k=10, dim=1, momentum=0.1, bias=True, eps=1e-05, beta=0.75, noise=False):
        super(GhostTopkBatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.register_buffer('meanTOPK', torch.zeros(num_features))
        self.noise = noise
        self.k = k
        self.beta = 0.75
        self.eps = eps
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = torch.abs(t.transpose(1, 0) - mean)
            beta = 0.75
            MeanTOPK = torch.topk(A, self.k, dim=0)[0].mean(0)
            meanTOPK = beta * torch.autograd.variable.Variable(self.biasTOPK) + (1 - beta) * MeanTOPK
            const = 0.5 * (1 + (np.pi * np.log(4)) ** 0.5) / (2 * np.log(A.size(0))) ** 0.5
            meanTOPK = meanTOPK * const
            self.biasTOPK.copy_(meanTOPK.data)
            scale = 1 / (meanTOPK + self.eps)
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)
        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class L1BatchNorm2d(nn.Module):
    """docstring for L1BatchNorm2d."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, normalized=True, eps=1e-05, noise=False):
        super(L1BatchNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.noise = noise
        self.bias = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(num_features))
        self.eps = eps
        if normalized:
            self.weight_fix = (np.pi / 2) ** 0.5
        else:
            self.weight_fix = 1

    def forward(self, x):
        p = 1
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            Var = torch.abs(t.transpose(1, 0) - mean).mean(0)
            scale = (Var * self.weight_fix + self.eps) ** -1
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_var)
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)
        if self.noise and self.training:
            std = 0.1 * _std(x, self.dim).data
            ones = torch.ones_like(x.data)
            std_noise = Variable(torch.normal(ones, ones) * std)
            out = out * std_noise
        if self.weight is not None:
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


_DEFAULT_FLATTEN = 1, -1


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values, num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if qparams is None:
            assert num_bits is not None, 'either provide qparams of num_bits to quantize'
            qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)
        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -2.0 ** (num_bits - 1) if signed else 0.0
        qmax = qmin + 2.0 ** num_bits - 1.0
        scale = qparams.range / (qmax - qmin)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_()
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN, inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits

    def forward(self, input, qparams=None):
        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range, zero_point=self.running_zero_point, num_bits=self.num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize, stochastic=self.stochastic, inplace=self.inplace)
            return q_input


_DEFAULT_FLATTEN_GRAD = 0, -1


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, 'either provide qparams of num_bits to quantize'
                qparams = calculate_qparams(grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')
            grad_input = quantize(grad_output, num_bits=None, qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias, stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None, stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach() if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits, flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class RangeBN(nn.Module):

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-05, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits, inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)
            mean_min = y.min(-1)[0].mean(-1)
            mean = y.view(C, -1).mean(-1)
            scale_fix = 0.5 * 0.35 * (1 + (math.pi * math.log(4)) ** 0.5) / (2 * math.log(y.size(-1))) ** 0.5
            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        out = (x - mean.view(1, -1, 1, 1)) / (scale.view(1, -1, 1, 1) + self.eps)
        if self.weight is not None:
            qweight = self.weight
            out = out * qweight.view(1, -1, 1, 1)
        if self.bias is not None:
            qbias = self.bias
            out = out + qbias.view(1, -1, 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-05, num_bits=8, num_bits_grad=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum, affine, num_chunks, eps, num_bits, num_bits_grad)
        self.quantize_input = QuantMeasure(self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))


class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU(True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(nn.Linear(in_channels, in_channels // ratio), nn.ReLU(inplace=True), nn.Linear(in_channels // ratio, out_channels), nn.Sigmoid())

    def forward(self, x):
        x_avg = self.global_pool(x).flatten(1, -1)
        mask = self.transform(x_avg)
        return x * mask.unsqueeze(-1).unsqueeze(-1)


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None, mixup=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * expansion))
        if residual_block is not None:
            residual_block = residual_block(out_planes)
        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion, downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups, residual_block=residual_block, dropout=dropout))
        if mixup:
            layers.append(MixUp())
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def linear_scale(lr0, lrT, T, t0=0):
    rate = (lrT - lr0) / T
    return "lambda t: {'lr': max(%s + (t - %s) * %s, 0)}" % (lr0, t0, rate)


def mixsize_config(sz, base_size, base_batch, base_duplicates, adapt_batch, adapt_duplicates):
    assert adapt_batch or adapt_duplicates or sz == base_size
    batch_size = base_batch
    duplicates = base_duplicates
    if adapt_batch and adapt_duplicates:
        scale = base_size / sz
    else:
        scale = (base_size / sz) ** 2
    if scale * duplicates < 0.5:
        adapt_duplicates = False
        adapt_batch = True
    if adapt_batch:
        batch_size = int(round(scale * base_batch))
    if adapt_duplicates:
        duplicates = int(round(scale * duplicates))
    duplicates = max(1, duplicates)
    return {'input_size': sz, 'batch_size': batch_size, 'duplicates': duplicates}


class ResNet_imagenet(ResNet):
    num_train_images = 1281167

    def __init__(self, num_classes=1000, inplanes=64, block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3], width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1], regime='normal', scale_lr=1, ramp_up_lr=True, ramp_up_epochs=5, checkpoint_segments=0, mixup=False, epochs=90, base_devices=4, base_device_batch=64, base_duplicates=1, base_image_size=224, mix_size_regime='D+'):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion, stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i], mixup=mixup)
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1] * expansion, num_classes)
        init_model(self)
        batch_size = base_devices * base_device_batch
        num_steps_epoch = math.floor(self.num_train_images / batch_size)
        ramp_up_steps = num_steps_epoch * ramp_up_epochs
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': scale_lr * 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
        if 'cutmix' in regime:
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': scale_lr * 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 75, 'lr': scale_lr * 0.01}, {'epoch': 150, 'lr': scale_lr * 0.001}, {'epoch': 225, 'lr': scale_lr * 0.0001}]
        if 'linear' in regime:
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': scale_lr * 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001), 'step_lambda': linear_scale(scale_lr * 0.1, 0, num_steps_epoch * epochs)}]
            if ramp_up_lr:
                self.regime[0]['lr'] = 0
                self.regime['step_lambda'] = linear_scale(0.1, scale_lr * 0.1, ramp_up_steps)
                self.regime.append({'epoch': ramp_up_epochs, 'step_lambda': linear_scale(scale_lr * 0.1, 0, num_steps_epoch * (epochs - ramp_up_epochs), ramp_up_steps)})
                ramp_up_lr = False
        if 'sampled' in regime:
            self.regime[0]['regularizer'] = [{'name': 'GradSmooth', 'momentum': 0.9, 'log': False}, weight_decay_config(0.0001)]
            ramp_up_lr = False
            self.data_regime = None

            def size_config(size):
                return mixsize_config(size, base_size=base_image_size, base_batch=base_device_batch, base_duplicates=base_duplicates, adapt_batch=mix_size_regime == 'B+', adapt_duplicates=mix_size_regime == 'D+')
            increment = int(base_image_size / 7)
            if '144' in regime:
                self.sampled_data_regime = [(0.1, size_config(base_image_size + increment)), (0.1, size_config(base_image_size)), (0.6, size_config(base_image_size - 3 * increment)), (0.2, size_config(base_image_size - 4 * increment))]
            else:
                self.sampled_data_regime = [(0.8 / 6, size_config(base_image_size - 3 * increment)), (0.8 / 6, size_config(base_image_size - 2 * increment)), (0.8 / 6, size_config(base_image_size - increment)), (0.2, size_config(base_image_size)), (0.8 / 6, size_config(base_image_size + increment)), (0.8 / 6, size_config(base_image_size + 2 * increment)), (0.8 / 6, size_config(base_image_size + 3 * increment))]
            self.data_eval_regime = [{'epoch': 0, 'input_size': base_image_size}]
        if ramp_up_lr and scale_lr > 1:
            self.regime[0]['step_lambda'] = linear_scale(0.1, 0.1 * scale_lr, ramp_up_steps)
            self.regime.insert(1, {'epoch': ramp_up_epochs, 'lr': scale_lr * 0.1})


class ResNet_cifar(ResNet):

    def __init__(self, num_classes=10, inplanes=16, block=BasicBlock, depth=18, width=[16, 32, 64], groups=[1, 1, 1], residual_block=None, regime='normal', dropout=None, mixup=False):
        super(ResNet_cifar, self).__init__()
        self.inplanes = inplanes
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, width[0], n, groups=groups[0], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer2 = self._make_layer(block, width[1], n, stride=2, groups=groups[1], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer3 = self._make_layer(block, width[2], n, stride=2, groups=groups[2], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1], num_classes)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 81, 'lr': 0.01}, {'epoch': 122, 'lr': 0.001}, {'epoch': 164, 'lr': 0.0001}]
        if 'wide-resnet' in regime:
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0005)}, {'epoch': 60, 'lr': 0.02}, {'epoch': 120, 'lr': 0.004}, {'epoch': 160, 'lr': 0.0008}]
        if 'sampled' in regime:
            adapt_batch = True if 'B+' in regime else False
            adapt_duplicates = True if 'D+' in regime or not adapt_batch else False

            def size_config(size):
                return mixsize_config(size, base_size=32, base_batch=64, base_duplicates=1, adapt_batch=adapt_batch, adapt_duplicates=adapt_duplicates)
            self.regime[0]['regularizer'] = [{'name': 'GradSmooth', 'momentum': 0.9, 'log': False}, weight_decay_config(0.0001)]
            self.data_regime = None
            self.sampled_data_regime = [(0.3, size_config(32)), (0.2, size_config(48)), (0.3, size_config(24)), (0.2, size_config(16))]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 32, 'scale_size': 32}]


class ResNetZI(nn.Module):

    def __init__(self):
        super(ResNetZI, self).__init__()
        self.num_layers = 1

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = ZIConv2d(self.inplanes, out_planes, kernel_size=1, stride=stride, bias=False)
        if residual_block is not None:
            residual_block = residual_block(out_planes)
        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion, downsample=downsample, groups=groups, residual_block=residual_block, layer_depth=self.num_layers))
        self.inplanes = planes * expansion
        self.num_layers += 1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups, residual_block=residual_block, layer_depth=self.num_layers))
            self.num_layers += 1
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class ResNetZI_imagenet(ResNetZI):

    def __init__(self, num_classes=1000, inplanes=64, block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3], width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1], regime='normal', scale_lr=1, checkpoint_segments=0):
        super(ResNetZI_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = ZIConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion, stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i])
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = ZILinear(width[-1] * expansion, num_classes, bias=True, post_bias=False)
        init_model(self)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001), 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)}, {'epoch': 5, 'lr': scale_lr * 0.1}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
        elif regime == 'fast':
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001), 'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))}, {'epoch': 4, 'lr': 4 * scale_lr * 0.1}, {'epoch': 18, 'lr': scale_lr * 0.1}, {'epoch': 21, 'lr': scale_lr * 0.01}, {'epoch': 35, 'lr': scale_lr * 0.001}, {'epoch': 43, 'lr': scale_lr * 0.0001}]
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 256}, {'epoch': 18, 'input_size': 224, 'batch_size': 64}, {'epoch': 41, 'input_size': 288, 'batch_size': 32}]
        elif regime == 'small':
            scale_lr *= 4
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(0.0001), 'momentum': 0.9, 'lr': scale_lr * 0.1}, {'epoch': 30, 'lr': scale_lr * 0.01}, {'epoch': 60, 'lr': scale_lr * 0.001}, {'epoch': 80, 'lr': scale_lr * 0.0001}]
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 256}, {'epoch': 80, 'input_size': 224, 'batch_size': 64}]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 1024}, {'epoch': 80, 'input_size': 224, 'batch_size': 512}]
        elif regime == 'small_ba':
            scale_lr = 1
            self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(0.0001), 'momentum': 0.9, 'lr': scale_lr * 0.1}, {'epoch': 20, 'lr': scale_lr * 0.01}, {'epoch': 25, 'lr': scale_lr * 0.001}, {'epoch': 28, 'lr': scale_lr * 0.0001}]
            self.data_regime = [{'epoch': 0, 'input_size': 128, 'batch_size': 64, 'duplicates': 4}, {'epoch': 25, 'input_size': 224, 'batch_size': 64, 'duplicates': 1}]
            self.data_eval_regime = [{'epoch': 0, 'input_size': 224, 'batch_size': 128}]


class ResNetZI_cifar(ResNetZI):

    def __init__(self, num_classes=10, inplanes=16, block=BasicBlock, depth=18, width=[16, 32, 64], groups=[1, 1, 1], residual_block=None):
        super(ResNetZI_cifar, self).__init__()
        self.inplanes = inplanes
        n = int((depth - 2) / 6)
        self.conv1 = ZIConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, width[0], n, groups=groups[0], residual_block=residual_block)
        self.layer2 = self._make_layer(block, width[1], n, stride=2, groups=groups[1], residual_block=residual_block)
        self.layer3 = self._make_layer(block, width[2], n, stride=2, groups=groups[2], residual_block=residual_block)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = ZILinear(width[-1], num_classes, bias=True, post_bias=False)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'regularizer': weight_decay_config(0.0001)}, {'epoch': 81, 'lr': 0.01}, {'epoch': 122, 'lr': 0.001}, {'epoch': 164, 'lr': 0.0001}]


class ResNeXt_imagenet(ResNet_imagenet):

    def __init__(self, width=[128, 256, 512, 1024], groups=[32, 32, 32, 32], expansion=2, **kwargs):
        kwargs['width'] = width
        kwargs['groups'] = groups
        kwargs['expansion'] = expansion
        super(ResNeXt_imagenet, self).__init__(**kwargs)


class ResNeXt_cifar(ResNet_cifar):

    def __init__(self, width=[64, 128, 256], groups=[4, 8, 16], **kwargs):
        kwargs['width'] = width
        kwargs['groups'] = groups
        super(ResNeXt_cifar, self).__init__(**kwargs)


class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes=10, batch_norm=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = self.classifier = nn.Sequential(nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0005, 'momentum': 0.9}, {'epoch': 60, 'lr': 0.1 * 0.2 ** 1}, {'epoch': 120, 'lr': 0.1 * 0.2 ** 2}, {'epoch': 160, 'lr': 0.1 * 0.2 ** 3}]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNetOWT_BN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (AmoebaNetCell,
     lambda: ([], {'C_prev_prev': 4, 'C_prev': 4, 'C': 4, 'reduction': 4, 'reduction_prev': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (BiReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBNAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DARTSCell,
     lambda: ([], {'C_prev_prev': 4, 'C_prev': 4, 'C': 4, 'reduction': 4, 'reduction_prev': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DepthwiseSeparableFusedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EvolvedNetworkCIFAR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EvolvedNetworkImageNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (ExpandedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostTopkBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HadamardProj,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionModule,
     lambda: ([], {'in_channels': 4, 'n1x1_channels': 4, 'n3x3r_channels': 4, 'n3x3_channels': 4, 'dn3x3r_channels': 4, 'dn3x3_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearFixed,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LpBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MBConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MBConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NasNetCell,
     lambda: ([], {'C_prev_prev': 4, 'C_prev': 4, 'C': 4, 'reduction': 4, 'reduction_prev': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Proj,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantMeasure,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RangeBN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RangeBN1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetZI_imagenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TopkBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ZIConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ZILinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (block17,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (block35,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (block8,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (mnist_model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_eladhoffer_convNet_pytorch(_paritybench_base):
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

