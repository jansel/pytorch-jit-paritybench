import sys
_module = sys.modules[__name__]
del sys
data_process = _module
data_loader = _module
transform = _module
jigsaw = _module
handcraft_ruls_postprocessing = _module
include = _module
jigsaw_puzzles = _module
loss = _module
bce_losses = _module
cyclic_lr = _module
lovasz_losses = _module
metric = _module
model = _module
ibnnet = _module
model = _module
senet = _module
predict = _module
prepare_data = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import random


import numpy as np


import torch.optim as optim


from torch.autograd import Variable


import torch.nn as nn


from functools import partial


import math


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn.functional as F


from torch.nn import init


import torchvision


from collections import OrderedDict


from torch.utils import model_zoo


import time


import logging


from itertools import chain


from collections import Iterable


from math import isnan


class DiceLoss(nn.Module):

    def __init__(self, smooth=0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


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


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv1.weight.data.normal_(0, math.sqrt(2.0 / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, 1, None, ibn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)


def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)


def conv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)


def conv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)


class ConvRelu(nn.Module):

    def __init__(self, in_, out, kernel_size, norm_type=None):
        super(ConvRelu, self).__init__()
        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False
        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True
        if kernel_size == 3:
            self.conv = conv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = conv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = conv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = conv1x1(in_, out, is_bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x


class ImprovedIBNaDecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(ImprovedIBNaDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = IBN(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, int(channel / reduction), bias=False), nn.ReLU(inplace=True), nn.Linear(int(channel / reduction), channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x, e], 1)
            x = F.dropout2d(x, p=0.5)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x


class Decoder_bottleneck(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder_bottleneck, self).__init__()
        self.block1 = Bottleneck(in_channels, channels)
        self.block2 = Bottleneck(channels, out_channels)
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.SCSE(x)
        return x


class model34_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1, mask_class=2):
        super(model34_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)
        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)
        hypercol = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear'), F.upsample(d3, scale_factor=4, mode='bilinear'), F.upsample(d4, scale_factor=8, mode='bilinear'), F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
        hypercol = F.dropout2d(hypercol, p=0.5)
        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((hypercol, F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(hypercol_add_center)
        return center_fc, x_no_empty, x_final


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
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
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


pretrained_settings = {'senet154': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet50': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet101': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnet152': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnext50_32x4d': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}, 'se_resnext101_32x4d': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth', 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'num_classes': 1000}}}


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class model50A_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model50A_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.decoder5 = Decoder(256 + 512 * 4, 512, 64)
        self.decoder4 = Decoder(64 + 256 * 4, 256, 64)
        self.decoder3 = Decoder(64 + 128 * 4, 128, 64)
        self.decoder2 = Decoder(64 + 64 * 4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)
        hypercol = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear'), F.upsample(d3, scale_factor=4, mode='bilinear'), F.upsample(d4, scale_factor=8, mode='bilinear'), F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
        hypercol = F.dropout2d(hypercol, p=0.5)
        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((hypercol, F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(hypercol_add_center)
        return center_fc, x_no_empty, x_final


class model50A_slim_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model50A_slim_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)
        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)
        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)
        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)
        self.decoder1 = Decoder_bottleneck(64, 32, 64)
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        f = self.center(conv5)
        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)
        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)
        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)
        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)
        hypercol = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear'), F.upsample(d3, scale_factor=4, mode='bilinear'), F.upsample(d4, scale_factor=8, mode='bilinear'), F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
        hypercol = F.dropout2d(hypercol, p=0.5)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)
        hypercol_add_center = torch.cat((hypercol, F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)
        return center_fc, x_no_empty, x_final


def state_dict_remove_moudle(moudle_state_dict, model):
    state_dict = model.state_dict()
    keys = list(moudle_state_dict.keys())
    for key in keys:
        None
        new_key = key.replace('module.', '')
        None
        state_dict[new_key] = moudle_state_dict[key]
    return state_dict


def resnext101_ibn_a(baseWidth, cardinality, pretrained=True):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 23, 3], 1000)
    if pretrained:
        state_dict = torch.load('/data/shentao/Airbus/code/pretrained_model/resnext101_ibn_a.pth.tar')['state_dict']
        state_dict = state_dict_remove_moudle(state_dict, model)
        model.load_state_dict(state_dict)
    return model


class model101A_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model101A_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        num_filters = 32
        baseWidth = 4
        cardinality = 32
        self.encoder = resnext101_ibn_a(baseWidth, cardinality, pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center_se = SELayer(512 * 4)
        self.center = ImprovedIBNaDecoderBlock(512 * 4, num_filters * 8)
        self.dec5_se = SELayer(512 * 4 + num_filters * 8)
        self.dec5 = ImprovedIBNaDecoderBlock(512 * 4 + num_filters * 8, num_filters * 8)
        self.dec4_se = SELayer(256 * 4 + num_filters * 8)
        self.dec4 = ImprovedIBNaDecoderBlock(256 * 4 + num_filters * 8, num_filters * 8)
        self.dec3_se = SELayer(128 * 4 + num_filters * 8)
        self.dec3 = ImprovedIBNaDecoderBlock(128 * 4 + num_filters * 8, num_filters * 4)
        self.dec2_se = SELayer(64 * 4 + num_filters * 4)
        self.dec2 = ImprovedIBNaDecoderBlock(64 * 4 + num_filters * 4, num_filters * 4)
        self.logits_no_empty = nn.Sequential(StConvRelu(num_filters * 4, num_filters, 3), nn.Dropout2d(0.5), nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(StConvRelu(num_filters * 4 + 64, num_filters, 3), nn.Dropout2d(0.5), nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_2048 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_2048)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        center = self.center(self.center_se(self.pool(conv5)))
        dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1)))
        dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))
        dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))
        dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))
        x_no_empty = self.logits_no_empty(dec2)
        dec0_add_center = torch.cat((dec2, F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(dec0_add_center)
        return center_fc, x_no_empty, x_final


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class model101B_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model101B_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        num_filters = 32
        self.encoder = se_resnext101_32x4d()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center_se = SELayer(512 * 4)
        self.center = ImprovedIBNaDecoderBlock(512 * 4, num_filters * 8)
        self.dec5_se = SELayer(512 * 4 + num_filters * 8)
        self.dec5 = ImprovedIBNaDecoderBlock(512 * 4 + num_filters * 8, num_filters * 8)
        self.dec4_se = SELayer(256 * 4 + num_filters * 8)
        self.dec4 = ImprovedIBNaDecoderBlock(256 * 4 + num_filters * 8, num_filters * 8)
        self.dec3_se = SELayer(128 * 4 + num_filters * 8)
        self.dec3 = ImprovedIBNaDecoderBlock(128 * 4 + num_filters * 8, num_filters * 4)
        self.dec2_se = SELayer(64 * 4 + num_filters * 4)
        self.dec2 = ImprovedIBNaDecoderBlock(64 * 4 + num_filters * 4, num_filters * 4)
        self.logits_no_empty = nn.Sequential(ConvRelu(num_filters * 4, num_filters, 3), nn.Dropout2d(0.5), nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(ConvRelu(num_filters * 4 + 64, num_filters, 3), nn.Dropout2d(0.5), nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_2048 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_2048)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        center = self.center(self.center_se(self.pool(conv5)))
        dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1)))
        dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))
        dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))
        dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))
        x_no_empty = self.logits_no_empty(dec2)
        dec0_add_center = torch.cat((dec2, F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(dec0_add_center)
        return center_fc, x_no_empty, x_final


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class model152_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model152_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        self.encoder = se_resnet152()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)
        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)
        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)
        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)
        self.decoder1 = Decoder_bottleneck(64, 32, 64)
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        f = self.center(conv5)
        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)
        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)
        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)
        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)
        hypercol = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear'), F.upsample(d3, scale_factor=4, mode='bilinear'), F.upsample(d4, scale_factor=8, mode='bilinear'), F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
        hypercol = F.dropout2d(hypercol, p=0.5)
        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((hypercol, F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
        x_final = self.logits_final(hypercol_add_center)
        return center_fc, x_no_empty, x_final


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class model154_DeepSupervion(nn.Module):

    def __init__(self, num_classes=1):
        super(model154_DeepSupervion, self).__init__()
        self.num_classes = num_classes
        self.encoder = senet154()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.encoder.layer0.conv2, self.encoder.layer0.bn2, self.encoder.layer0.relu2, self.encoder.layer0.conv3, self.encoder.layer0.bn3, self.encoder.layer0.relu3)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)
        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)
        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)
        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)
        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)
        self.decoder1 = Decoder_bottleneck(64, 32, 64)
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)
        f = self.center(conv5)
        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)
        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)
        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)
        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)
        hypercol = torch.cat((d1, F.upsample(d2, scale_factor=2, mode='bilinear'), F.upsample(d3, scale_factor=4, mode='bilinear'), F.upsample(d4, scale_factor=8, mode='bilinear'), F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
        hypercol = F.dropout2d(hypercol, p=0.5)
        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((hypercol, F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
        x_final = self.logits_final(hypercol_add_center)
        return center_fc, x_no_empty, x_final


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'out_channels': 16}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SCSEBlock,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SELayer,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (model34_DeepSupervion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (model50A_DeepSupervion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
]

class Test_SeuTao_TGS_Salt_Identification_Challenge_2018_4th_place_solution(_paritybench_base):
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

