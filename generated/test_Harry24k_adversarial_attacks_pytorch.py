import sys
_module = sys.modules[__name__]
del sys
collect_env = _module
test_atks = _module
test_import = _module
utils = _module
conf = _module
robustbench = _module
data = _module
eval = _module
leaderboard = _module
template = _module
loaders = _module
model_zoo = _module
CARD_resnet = _module
architectures = _module
boosting_wide_resnet = _module
dm_wide_resnet = _module
paf_wide_resnet = _module
resnest = _module
resnet = _module
resnext = _module
robust_wide_resnet = _module
utils_architectures = _module
wide_resnet = _module
cifar10 = _module
cifar100 = _module
enums = _module
imagenet = _module
models = _module
utils = _module
zenodo_download = _module
torchattacks = _module
attack = _module
attacks = _module
_differential_evolution = _module
apgd = _module
apgdt = _module
autoattack = _module
bim = _module
cw = _module
deepfool = _module
difgsm = _module
eotpgd = _module
fab = _module
ffgsm = _module
fgsm = _module
gn = _module
jitter = _module
mifgsm = _module
nifgsm = _module
onepixel = _module
pgd = _module
pgdl2 = _module
pgdrs = _module
pgdrsl2 = _module
pixle = _module
rfgsm = _module
sinifgsm = _module
sparsefool = _module
square = _module
tifgsm = _module
tpgd = _module
upgd = _module
vanila = _module
vmifgsm = _module
vnifgsm = _module
wrappers = _module
lgv = _module
multiattack = _module

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


import itertools


import time


import torch


import numpy as np


import matplotlib.pyplot as plt


import torchvision


import torchvision.datasets as dsets


import torchvision.transforms as transforms


from typing import Callable


from typing import Union


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import Set


from typing import Tuple


import torch.utils.data as data


import torchvision.datasets as datasets


from torch.utils.data import Dataset


from torchvision.datasets.vision import VisionDataset


import math


import torch.nn.functional as F


from torch import nn


from typing import Type


import torch.nn as nn


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


from torch.nn import init


from collections import OrderedDict


from typing import TypeVar


from torch import Tensor


import warnings


import logging


from collections.abc import Iterable


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import torch.optim as optim


from collections import abc as container_abcs


import copy


from itertools import chain


from torch.nn.functional import softmax


from scipy import stats as st


from random import shuffle


from random import sample


class PreActBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, affine=False)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), torch.nn.BatchNorm2d(self.expansion * planes, affine=False))

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        out += shortcut
        return out


class WidePreActResNet(torch.nn.Module):

    def __init__(self, block=PreActBasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, widen_factor=2):
        super(WidePreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256 * (widen_factor + 1) * block.expansion, affine=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64 * (widen_factor + 1), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128 * (widen_factor + 1), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256 * (widen_factor + 1), num_blocks[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(256 * (widen_factor + 1) * block.expansion, num_classes, kernel_size=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.relu(self.bn1(out))
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


class _Swish(torch.autograd.Function):
    """Custom implementation of swish."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Module using custom implementation."""

    def forward(self, input_tensor):
        return _Swish.apply(input_tensor)


class _Block(nn.Module):
    """WideResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn: Type[nn.Module]=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """WideResNet block group."""

    def __init__(self, num_blocks, in_planes, out_planes, stride, activation_fn: Type[nn.Module]=nn.ReLU):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(_Block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, activation_fn=activation_fn))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


CIFAR10_MEAN = 0.4914, 0.4822, 0.4465


CIFAR10_STD = 0.2471, 0.2435, 0.2616


class DMWideResNet(nn.Module):
    """WideResNet."""

    def __init__(self, num_classes: int=10, depth: int=28, width: int=10, activation_fn: Type[nn.Module]=nn.ReLU, mean: Union[Tuple[float, ...], float]=CIFAR10_MEAN, std: Union[Tuple[float, ...], float]=CIFAR10_STD, padding: int=0, num_input_channels: int=3):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(num_input_channels, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(num_input_channels, 1, 1), persistent=False)
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(_BlockGroup(num_blocks, num_channels[0], num_channels[1], 1, activation_fn=activation_fn), _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2, activation_fn=activation_fn), _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2, activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


class _PreActBlock(nn.Module):
    """Pre-activation ResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self._stride = stride
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        self.conv_2d_1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_2d_2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.has_shortcut = stride != 1 or in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)

    def _pad(self, x):
        if self._stride == 1:
            x = F.pad(x, (1, 1, 1, 1))
        elif self._stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        return x

    def forward(self, x):
        out = self.relu_0(self.batchnorm_0(x))
        shortcut = self.shortcut(self._pad(x)) if self.has_shortcut else x
        out = self.conv_2d_1(self._pad(out))
        out = self.conv_2d_2(self.relu_1(self.batchnorm_1(out)))
        return out + shortcut


class DMPreActResNet(nn.Module):
    """Pre-activation ResNet."""

    def __init__(self, num_classes: int=10, depth: int=18, width: int=0, activation_fn: Type[nn.Module]=nn.ReLU, mean: Union[Tuple[float, ...], float]=CIFAR10_MEAN, std: Union[Tuple[float, ...], float]=CIFAR10_STD, padding: int=0, num_input_channels: int=3, use_cuda: bool=True):
        super().__init__()
        if width != 0:
            raise ValueError('Unsupported `width`.')
        self.register_buffer('mean', torch.tensor(mean).view(num_input_channels, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(num_input_channels, 1, 1), persistent=False)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        self.conv_2d = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if depth == 18:
            num_blocks = 2, 2, 2, 2
        elif depth == 34:
            num_blocks = 3, 4, 6, 3
        else:
            raise ValueError('Unsupported `depth`.')
        self.layer_0 = self._make_layer(64, 64, num_blocks[0], 1, activation_fn)
        self.layer_1 = self._make_layer(64, 128, num_blocks[1], 2, activation_fn)
        self.layer_2 = self._make_layer(128, 256, num_blocks[2], 2, activation_fn)
        self.layer_3 = self._make_layer(256, 512, num_blocks[3], 2, activation_fn)
        self.batchnorm = nn.BatchNorm2d(512)
        self.relu = activation_fn()
        self.logits = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, activation_fn):
        layers = []
        for i, stride in enumerate([stride] + [1] * (num_blocks - 1)):
            layers.append(_PreActBlock(i == 0 and in_planes or out_planes, out_planes, stride, activation_fn))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.conv_2d(out)
        out = self.layer_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.logits(out)


class PSSiLU(nn.Module):

    def __init__(self):
        super(PSSiLU, self).__init__()
        self.beta = nn.Parameter(torch.tensor([1e-08]))
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x * (F.sigmoid(torch.abs(self.alpha) * x) - torch.abs(self.beta)) / (1 - torch.abs(self.beta))


class PAF_BasicBlock(nn.Module):

    def __init__(self, activation, in_planes, out_planes, stride, dropRate=0.0):
        super(PAF_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.activation(self.bn1(x))
        else:
            out = self.activation(self.bn1(x))
        out = self.activation(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class PAF_NetworkBlock(nn.Module):

    def __init__(self, activation, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(PAF_NetworkBlock, self).__init__()
        self.layer = self._make_layer(activation, block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, activation, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(activation, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class PAF_WideResNet(nn.Module):

    def __init__(self, activation, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, **kwargs):
        super(PAF_WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = PAF_BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = PAF_NetworkBlock(activation, n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = PAF_NetworkBlock(activation, n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = PAF_NetworkBlock(activation, n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.activation = activation
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
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


class SplAtConv2d(Module):
    """Split-Attention Conv2d"""

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, norm_layer=None, dropblock_prob=0.0, swish=False, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = nn.SiLU() if swish else nn.ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNest(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64, num_classes=10, dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0, last_gamma=False, norm_layer=nn.BatchNorm2d, swish=False):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.swish = swish
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.relu = nn.SiLU if swish else nn.ReLU
        super(ResNest, self).__init__()
        conv_layer = nn.Conv2d
        conv_kwargs = {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs), norm_layer(stem_width), self.relu(), conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs), norm_layer(stem_width), self.relu(), conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=3, bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = self.relu()
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma, swish=self.swish))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma, swish=self.swish))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=dilation, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma, swish=self.swish))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BottleneckChen2020AdversarialNet(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckChen2020AdversarialNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        pre = F.relu(self.bn0(x))
        out = F.relu(self.bn1(self.conv1(pre)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if len(self.shortcut) == 0:
            out += self.shortcut(x)
        else:
            out += self.shortcut(pre)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_shortcut=False):
        super(PreActBlock, self).__init__()
        self.out_shortcut = out_shortcut
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out if self.out_shortcut else x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBlockV2(nn.Module):
    """Pre-activation version of the BasicBlock (slightly different forward pass)"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_shortcut=False):
        super(PreActBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, bn_before_fc=False, out_shortcut=False):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.bn_before_fc = bn_before_fc
        self.out_shortcut = out_shortcut
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if bn_before_fc:
            self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, out_shortcut=self.out_shortcut))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.bn_before_fc:
            out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua).
    """
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        dim = int(math.floor(planes * (base_width / 64.0)))
        self.conv_reduce = nn.Conv2d(inplanes, dim * cardinality, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(dim * cardinality)
        self.conv_conv = nn.Conv2d(dim * cardinality, dim * cardinality, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(dim * cardinality)
        self.conv_expand = nn.Conv2d(dim * cardinality, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, num_classes):
        super(CifarResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RobustWideResNet(nn.Module):

    def __init__(self, num_classes=10, channel_configs=[16, 160, 320, 640], depth_configs=[5, 5, 5], stride_config=[1, 2, 2], drop_rate_config=[0.0, 0.0, 0.0]):
        super(RobustWideResNet, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config) == len(drop_rate_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = nn.ModuleList([])
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(block=BasicBlock, nb_layers=depth_configs[i], in_planes=channel_configs[i], out_planes=channel_configs[i + 1], stride=stride, dropRate=drop_rate_config[i]))
        self.bn1 = nn.BatchNorm2d(channel_configs[-1])
        self.relu = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_configs[-1], num_classes)
        self.fc_size = channel_configs[-1]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        return out


class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) ->None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) ->Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """

    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):

    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck, depth=depth, num_classes=100, cardinality=cardinality, base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Hendrycks2020AugMixWRNNet(WideResNet):

    def __init__(self, depth=40, widen_factor=2):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=False, num_classes=100)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Hendrycks2019UsingNet(WideResNet):

    def __init__(self, depth=28, widen_factor=10):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth, widen_factor=widen_factor, num_classes=100, sub_block1=False)

    def forward(self, x):
        x = 2.0 * x - 1.0
        return super(Hendrycks2019UsingNet, self).forward(x)


class Rice2020OverfittingNet(PreActResNet):

    def __init__(self):
        super(Rice2020OverfittingNet, self).__init__(PreActBlock, [2, 2, 2, 2], num_classes=100, bn_before_fc=True, out_shortcut=True)
        self.register_buffer('mu', torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Engstrom2019RobustnessNet(ResNet):

    def __init__(self):
        super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2023, 0.1994, 0.201]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Engstrom2019RobustnessNet, self).forward(x)


class Chen2020AdversarialNet(nn.Module):

    def __init__(self):
        super(Chen2020AdversarialNet, self).__init__()
        self.branch1 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch2 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch3 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.models = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = (x - self.mu) / self.sigma
        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)
        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
        prob3 = torch.softmax(out3, dim=1)
        return (prob1 + prob2 + prob3) / 3


class Wong2020FastNet(PreActResNet):

    def __init__(self):
        super(Wong2020FastNet, self).__init__(PreActBlock, [2, 2, 2, 2])
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Wong2020FastNet, self).forward(x)


class Ding2020MMANet(WideResNet):
    """
    See the appendix of the LICENSE file specifically for this model.
    """

    def __init__(self, depth=28, widen_factor=4):
        super(Ding2020MMANet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        std_min = torch.ones_like(std) / (x.shape[1] * x.shape[2] * x.shape[3]) ** 0.5
        x = (x - mu) / torch.max(std, std_min)
        return super(Ding2020MMANet, self).forward(x)


class Augustin2020AdversarialNet(ResNet):

    def __init__(self):
        super(Augustin2020AdversarialNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer('mu', torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.24703225141799082, 0.24348516474564, 0.2615878392604963]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Augustin2020AdversarialNet, self).forward(x)


class Augustin2020AdversarialWideNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super(Augustin2020AdversarialWideNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)
        self.register_buffer('mu', torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.24703225141799082, 0.24348516474564, 0.2615878392604963]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Augustin2020AdversarialWideNet, self).forward(x)


class Rice2020OverfittingNetL2(PreActResNet):

    def __init__(self):
        super(Rice2020OverfittingNetL2, self).__init__(PreActBlockV2, [2, 2, 2, 2], bn_before_fc=True)
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNetL2, self).forward(x)


class Rony2019DecouplingNet(WideResNet):

    def __init__(self, depth=28, widen_factor=10):
        super(Rony2019DecouplingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)
        self.register_buffer('mu', torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rony2019DecouplingNet, self).forward(x)


class Kireev2021EffectivenessNet(PreActResNet):

    def __init__(self):
        super(Kireev2021EffectivenessNet, self).__init__(PreActBlockV2, [2, 2, 2, 2], bn_before_fc=True)
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Kireev2021EffectivenessNet, self).forward(x)


class Chen2020EfficientNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=True, num_classes=100)
        self.register_buffer('mu', torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class LRR_ResNet(torchvision.models.ResNet):
    expansion = 1

    def __init__(self, block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=10, width=64):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""
        torch.nn.Module.__init__(self)
        self._norm_layer = torch.nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2, dilate=False)
        self.avgpool = torch.nn.AvgPool2d(4)
        self.fc = torch.nn.Linear(width * 8 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Diffenderfer2021CARD_Deck(torch.nn.Module):

    def __init__(self, width=128, num_classes=100):
        super(Diffenderfer2021CARD_Deck, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()
        for i in range(self.num_cards):
            self.models.append(LRR_ResNet(width=width, num_classes=num_classes))
        self.register_buffer('mu', torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        x_cl = x.clone()
        out_list = []
        for i in range(self.num_cards):
            out = self.models[i](x_cl)
            out = torch.softmax(out, dim=1)
            out_list.append(out)
        return torch.mean(torch.stack(out_list), dim=0)


class Diffenderfer2021CARD_Binary(WidePreActResNet):

    def __init__(self, num_classes=100):
        super(Diffenderfer2021CARD_Binary, self).__init__(num_classes=num_classes)
        self.register_buffer('mu', torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Diffenderfer2021CARD_Deck_Binary(torch.nn.Module):

    def __init__(self, num_classes=100):
        super(Diffenderfer2021CARD_Deck_Binary, self).__init__()
        self.num_cards = 6
        self.models = nn.ModuleList()
        for i in range(self.num_cards):
            self.models.append(WidePreActResNet(num_classes=num_classes))
        self.register_buffer('mu', torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        x_cl = x.clone()
        out_list = []
        for i in range(self.num_cards):
            out = self.models[i](x_cl)
            out = torch.softmax(out, dim=1)
            out_list.append(out)
        return torch.mean(torch.stack(out_list), dim=0)


class Modas2021PRIMEResNet18(ResNet):

    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Wu2020AdversarialNet(WideResNet):

    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=False, num_classes=100)
        self.register_buffer('mu', torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class LightEnsemble(nn.Module):

    def __init__(self, list_models, order='shuffle', n_grad=1):
        """
        Perform a single forward pass to one of the models when call forward()

        Arguments:
            list_models (list of nn.Module): list of LGV models.
            order (str): 'shuffle' draw a model without replacement (default), 'random' draw a model with replacement,
            None cycle in provided order.
            n_grad (int): number of models to ensemble in each forward pass (fused logits). Select models according to
            `order`. If equal to -1, use all models and order is ignored.
        """
        super(LightEnsemble, self).__init__()
        self.n_models = len(list_models)
        if self.n_models < 1:
            raise ValueError('Empty list of models')
        if not (n_grad > 0 or n_grad == -1):
            raise ValueError('n_grad should be strictly positive or equal to -1')
        if order == 'shuffle':
            shuffle(list_models)
        elif order in [None, 'random']:
            pass
        else:
            raise ValueError('Not supported order')
        self.models = nn.ModuleList(list_models)
        self.order = order
        self.n_grad = n_grad
        self.f_count = 0

    def forward(self, x):
        if self.n_grad >= self.n_models or self.n_grad < 0:
            indexes = list(range(self.n_models))
        elif self.order == 'random':
            indexes = sample(range(self.n_models), self.n_grad)
        else:
            indexes = [(i % self.n_models) for i in list(range(self.f_count, self.f_count + self.n_grad))]
            self.f_count += self.n_grad
        if self.n_grad == 1:
            x = self.models[indexes[0]](x)
        else:
            x_list = [model(x.clone()) for i, model in enumerate(self.models) if i in indexes]
            x = torch.stack(x_list)
            x = torch.mean(x, dim=0, keepdim=False)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Augustin2020AdversarialWideNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckChen2020AdversarialNet,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Chen2020EfficientNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DMWideResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Diffenderfer2021CARD_Binary,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Diffenderfer2021CARD_Deck_Binary,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Ding2020MMANet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hendrycks2019UsingNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Hendrycks2020AugMixWRNNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PAF_BasicBlock,
     lambda: ([], {'activation': _mock_layer(), 'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PAF_WideResNet,
     lambda: ([], {'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PSSiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBlockV2,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RobustWideResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Rony2019DecouplingNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SplAtConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WidePreActResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (WideResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Wu2020AdversarialNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_Block,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_BlockGroup,
     lambda: ([], {'num_blocks': 4, 'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_PreActBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (rSoftMax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Harry24k_adversarial_attacks_pytorch(_paritybench_base):
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

