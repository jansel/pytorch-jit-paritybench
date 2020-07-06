import sys
_module = sys.modules[__name__]
del sys
ptsemseg = _module
augmentations = _module
caffe_pb2 = _module
loader = _module
ade20k_loader = _module
camvid_loader = _module
cityscapes_loader = _module
mapillary_vistas_loader = _module
mit_sceneparsing_benchmark_loader = _module
nyuv2_loader = _module
pascal_voc_loader = _module
sunrgbd_loader = _module
loss = _module
loss = _module
metrics = _module
models = _module
fcn = _module
frrn = _module
icnet = _module
linknet = _module
pspnet = _module
refinenet = _module
segnet = _module
unet = _module
utils = _module
optimizers = _module
schedulers = _module
schedulers = _module
test = _module
train = _module
validate = _module

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


import collections


import torch


import torchvision


import numpy as np


import scipy.misc as m


from torch.utils import data


import scipy.io as io


from torchvision import transforms


import torch.nn.functional as F


import functools


import torch.nn as nn


from torch.autograd import Variable


import logging


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import ASGD


from torch.optim import Adamax


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import RMSprop


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import _LRScheduler


import scipy.misc as misc


import time


import random


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        input = F.interpolate(input, size=(ht, wt), mode='bilinear', align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, weight=weight, size_average=size_average, ignore_index=250)
    return loss


class fcn32s(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=100), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, self.n_classes, 1))
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        out = F.upsample(score, x.size()[2:])
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn16s(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=100), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, self.n_classes, 1))
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        out = F.upsample(score, x.size()[2:])
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[(range(in_channels)), (range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class fcn8s(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=True):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=100), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, self.n_classes, 1))
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2, bias=False)
            self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16, stride=8, bias=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0]))

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)
            score_pool3c = self.score_pool3(conv3)[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
            out = self.upscore8(score_pool3c + upscore_pool4)[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]]
            return out.contiguous()
        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, is_batchnorm=True):
        super(conv2DBatchNormRelu, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DGroupNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16):
        super(conv2DGroupNormRelu, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        self.cgr_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False, n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels, k_size=3, stride=1, padding=1, bias=False, n_groups=self.n_groups)
            self.conv2 = conv_unit(out_channels, out_channels, k_size=3, stride=1, padding=1, bias=False, n_groups=self.n_groups)
        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels, k_size=3, stride=1, padding=1, bias=False)
            self.conv2 = conv_unit(out_channels, out_channels, k_size=3, stride=1, padding=1, bias=False)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)
        x = self.conv_res(y_prime)
        upsample_size = torch.Size([(_s * self.scale) for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode='nearest')
        z_prime = z + x
        return y_prime, z_prime


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, is_batchnorm=True):
        super(conv2DBatchNorm, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16):
        super(conv2DGroupNorm, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False, n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False, n_groups=self.n_groups)
            self.conv2 = conv2DGroupNorm(channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False, n_groups=self.n_groups)
        else:
            self.conv1 = conv2DBatchNormRelu(channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False)
            self.conv2 = conv2DBatchNorm(channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


frrn_specs_dic = {'A': {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]], 'decoder': [[2, 192, 8], [2, 192, 4], [2, 48, 2]]}, 'B': {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]], 'decoder': [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]]}}


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323

    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=21, model_type='B', group_norm=False, n_groups=16):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)
        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48, kernel_size=3, strides=1, group_norm=self.group_norm, n_groups=self.n_groups))
            self.down_residual_units.append(RU(channels=48, kernel_size=3, strides=1, group_norm=self.group_norm, n_groups=self.n_groups))
        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)
        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0, stride=1, bias=False)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']
        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale, group_norm=self.group_norm, n_groups=self.n_groups))
            prev_channels = channels
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale, group_norm=self.group_norm, n_groups=self.n_groups))
            prev_channels = channels
        self.merge_conv = nn.Conv2d(prev_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False)
        self.classif_conv = nn.Conv2d(48, self.n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(3):
            x = self.up_residual_units[i](x)
        y = x
        z = self.split_conv(x)
        prev_channels = 48
        for n_blocks, channels, scale in self.encoder_frru_specs:
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels
        for n_blocks, channels, scale in self.decoder_frru_specs:
            upsample_size = torch.Size([(_s * 2) for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear', align_corners=True)
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_upsampled, z)
            prev_channels = channels
        x = torch.cat([F.upsample(y, scale_factor=2, mode='bilinear', align_corners=True), z], dim=1)
        x = self.merge_conv(x)
        for i in range(3):
            x = self.down_residual_units[i](x)
        x = self.classif_conv(x)
        return x


def get_interp_size(input, s_factor=1, z_factor=1):
    ori_h, ori_w = input.shape[2:]
    ori_h = (ori_h - 1) / s_factor + 1
    ori_w = (ori_w - 1) / s_factor + 1
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)
    resize_shape = int(ori_h), int(ori_w)
    return resize_shape


class cascadeFeatureFusion(nn.Module):

    def __init__(self, n_classes, low_in_channels, high_in_channels, out_channels, is_batchnorm=True):
        super(cascadeFeatureFusion, self).__init__()
        bias = not is_batchnorm
        self.low_dilated_conv_bn = conv2DBatchNorm(low_in_channels, out_channels, 3, stride=1, padding=2, bias=bias, dilation=2, is_batchnorm=is_batchnorm)
        self.low_classifier_conv = nn.Conv2d(int(low_in_channels), int(n_classes), kernel_size=1, padding=0, stride=1, bias=True, dilation=1)
        self.high_proj_conv_bn = conv2DBatchNorm(high_in_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x_low, x_high):
        x_low_upsampled = F.interpolate(x_low, size=get_interp_size(x_low, z_factor=2), mode='bilinear', align_corners=True)
        low_cls = self.low_classifier_conv(x_low_upsampled)
        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)
        return high_fused_fm, low_cls


icnet_specs = {'cityscapes': {'n_classes': 19, 'input_size': (1025, 2049), 'block_config': [3, 4, 6, 3]}}


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)
    if scale_weight is None:
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float())
    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, size_average=size_average)
    return loss


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', is_batchnorm=True):
        super(pyramidPooling, self).__init__()
        bias = not is_batchnorm
        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, is_batchnorm=is_batchnorm))
        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]
        if self.training or self.model_name != 'icnet':
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
        else:
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]
        if self.fusion_mode == 'cat':
            output_slices = [x]
            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
                output_slices.append(out)
            return torch.cat(output_slices, dim=1)
        else:
            pp_sum = x
            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
                pp_sum = pp_sum + out
            return pp_sum


class bottleNeckIdentifyPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()
        bias = not is_batchnorm
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=1, padding=dilation, bias=bias, dilation=dilation, is_batchnorm=is_batchnorm)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=1, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class bottleNeckPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckPSP, self).__init__()
        bias = not is_batchnorm
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=dilation, bias=bias, dilation=dilation, is_batchnorm=is_batchnorm)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class residualBlockPSP(nn.Module):

    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, include_range='all', is_batchnorm=True):
        super(residualBlockPSP, self).__init__()
        if dilation > 1:
            stride = 1
        layers = []
        if include_range in ['all', 'conv']:
            layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation, is_batchnorm=is_batchnorm))
        if include_range in ['all', 'identity']:
            for i in range(n_blocks - 1):
                layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class icnet(nn.Module):
    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545

    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow

    """

    def __init__(self, n_classes=19, block_config=[3, 4, 6, 3], input_size=(1025, 2049), version=None, is_batchnorm=True):
        super(icnet, self).__init__()
        bias = not is_batchnorm
        self.block_config = icnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = icnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = icnet_specs[version]['input_size'] if version is not None else input_size
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=32, padding=1, stride=2, bias=bias, is_batchnorm=is_batchnorm)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=32, k_size=3, n_filters=32, padding=1, stride=1, bias=bias, is_batchnorm=is_batchnorm)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=32, k_size=3, n_filters=64, padding=1, stride=1, bias=bias, is_batchnorm=is_batchnorm)
        self.res_block2 = residualBlockPSP(self.block_config[0], 64, 32, 128, 1, 1, is_batchnorm=is_batchnorm)
        self.res_block3_conv = residualBlockPSP(self.block_config[1], 128, 64, 256, 2, 1, include_range='conv', is_batchnorm=is_batchnorm)
        self.res_block3_identity = residualBlockPSP(self.block_config[1], 128, 64, 256, 2, 1, include_range='identity', is_batchnorm=is_batchnorm)
        self.res_block4 = residualBlockPSP(self.block_config[2], 256, 128, 512, 1, 2, is_batchnorm=is_batchnorm)
        self.res_block5 = residualBlockPSP(self.block_config[3], 512, 256, 1024, 1, 4, is_batchnorm=is_batchnorm)
        self.pyramid_pooling = pyramidPooling(1024, [6, 3, 2, 1], model_name='icnet', fusion_mode='sum', is_batchnorm=is_batchnorm)
        self.conv5_4_k1 = conv2DBatchNormRelu(in_channels=1024, k_size=1, n_filters=256, padding=0, stride=1, bias=bias, is_batchnorm=is_batchnorm)
        self.convbnrelu1_sub1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=32, padding=1, stride=2, bias=bias, is_batchnorm=is_batchnorm)
        self.convbnrelu2_sub1 = conv2DBatchNormRelu(in_channels=32, k_size=3, n_filters=32, padding=1, stride=2, bias=bias, is_batchnorm=is_batchnorm)
        self.convbnrelu3_sub1 = conv2DBatchNormRelu(in_channels=32, k_size=3, n_filters=64, padding=1, stride=2, bias=bias, is_batchnorm=is_batchnorm)
        self.classification = nn.Conv2d(128, self.n_classes, 1, 1, 0)
        self.cff_sub24 = cascadeFeatureFusion(self.n_classes, 256, 256, 128, is_batchnorm=is_batchnorm)
        self.cff_sub12 = cascadeFeatureFusion(self.n_classes, 128, 64, 128, is_batchnorm=is_batchnorm)
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        h, w = x.shape[2:]
        x_sub2 = F.interpolate(x, size=get_interp_size(x, s_factor=2), mode='bilinear', align_corners=True)
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)
        x_sub2 = self.res_block2(x_sub2)
        x_sub2 = self.res_block3_conv(x_sub2)
        x_sub4 = F.interpolate(x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode='bilinear', align_corners=True)
        x_sub4 = self.res_block3_identity(x_sub4)
        x_sub4 = self.res_block4(x_sub4)
        x_sub4 = self.res_block5(x_sub4)
        x_sub4 = self.pyramid_pooling(x_sub4)
        x_sub4 = self.conv5_4_k1(x_sub4)
        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.convbnrelu3_sub1(x_sub1)
        x_sub24, sub4_cls = self.cff_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.cff_sub12(x_sub24, x_sub1)
        x_sub12 = F.interpolate(x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode='bilinear', align_corners=True)
        x_sub4 = self.res_block3_identity(x_sub4)
        sub124_cls = self.classification(x_sub12)
        if self.training:
            return sub124_cls, sub24_cls, sub4_cls
        else:
            sub124_cls = F.interpolate(sub124_cls, size=get_interp_size(sub124_cls, z_factor=4), mode='bilinear', align_corners=True)
            return sub124_cls

    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData', 'Convolution']

        def _get_layer_params(layer, ltype):
            if ltype == 'BNData':
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]
            elif ltype in ['ConvolutionData', 'HoleConvolutionData', 'Convolution']:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]
            elif ltype == 'InnerProduct':
                raise Exception('Fully connected layers {}, not supported'.format(ltype))
            else:
                raise Exception('Unkown layer type {}'.format(ltype))
        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())
        layer_types = {}
        layer_params = {}
        for l in net.layer:
            lname = l.name
            ltype = l.type
            lbottom = l.bottom
            ltop = l.top
            if ltype in ltypes:
                None
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False
            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())
            None
            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))
            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                None
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_bn(conv_layer_name, bn_module):
            mean, var, gamma, beta = layer_params[conv_layer_name + '/bn']
            None
            bn_module.running_mean.copy_(torch.from_numpy(mean).view_as(bn_module.running_mean))
            bn_module.running_var.copy_(torch.from_numpy(var).view_as(bn_module.running_var))
            bn_module.weight.data.copy_(torch.from_numpy(gamma).view_as(bn_module.weight))
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            _transfer_conv(conv_layer_name, conv_module)
            if conv_layer_name + '/bn' in layer_params.keys():
                bn_module = mother_module[1]
                _transfer_bn(conv_layer_name, bn_module)

        def _transfer_residual(block_name, block):
            block_module, n_layers = block[0], block[1]
            prefix = block_name[:5]
            if 'bottleneck' in block_name or 'identity' not in block_name:
                bottleneck = block_module.layers[0]
                bottleneck_conv_bn_dic = {(prefix + '_1_1x1_reduce'): bottleneck.cbr1.cbr_unit, (prefix + '_1_3x3'): bottleneck.cbr2.cbr_unit, (prefix + '_1_1x1_proj'): bottleneck.cb4.cb_unit, (prefix + '_1_1x1_increase'): bottleneck.cb3.cb_unit}
                for k, v in bottleneck_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)
            if 'identity' in block_name or 'bottleneck' not in block_name:
                base_idx = 2 if 'identity' in block_name else 1
                for layer_idx in range(2, n_layers + 1):
                    residual_layer = block_module.layers[layer_idx - base_idx]
                    residual_conv_bn_dic = {'_'.join(map(str, [prefix, layer_idx, '1x1_reduce'])): residual_layer.cbr1.cbr_unit, '_'.join(map(str, [prefix, layer_idx, '3x3'])): residual_layer.cbr2.cbr_unit, '_'.join(map(str, [prefix, layer_idx, '1x1_increase'])): residual_layer.cb3.cb_unit}
                    for k, v in residual_conv_bn_dic.items():
                        _transfer_conv_bn(k, v)
        convbn_layer_mapping = {'conv1_1_3x3_s2': self.convbnrelu1_1.cbr_unit, 'conv1_2_3x3': self.convbnrelu1_2.cbr_unit, 'conv1_3_3x3': self.convbnrelu1_3.cbr_unit, 'conv1_sub1': self.convbnrelu1_sub1.cbr_unit, 'conv2_sub1': self.convbnrelu2_sub1.cbr_unit, 'conv3_sub1': self.convbnrelu3_sub1.cbr_unit, 'conv5_4_k1': self.conv5_4_k1.cbr_unit, 'conv_sub4': self.cff_sub24.low_dilated_conv_bn.cb_unit, 'conv3_1_sub2_proj': self.cff_sub24.high_proj_conv_bn.cb_unit, 'conv_sub2': self.cff_sub12.low_dilated_conv_bn.cb_unit, 'conv3_sub1_proj': self.cff_sub12.high_proj_conv_bn.cb_unit}
        residual_layers = {'conv2': [self.res_block2, self.block_config[0]], 'conv3_bottleneck': [self.res_block3_conv, self.block_config[1]], 'conv3_identity': [self.res_block3_identity, self.block_config[1]], 'conv4': [self.res_block4, self.block_config[2]], 'conv5': [self.res_block5, self.block_config[3]]}
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)
        _transfer_conv('conv6_cls', self.classification)
        _transfer_conv('conv6_sub4', self.cff_sub24.low_classifier_conv)
        _transfer_conv('conv6_sub2', self.cff_sub12.low_classifier_conv)
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

    def tile_predict(self, imgs, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.

        Strides are adaptively computed from the imgs shape
        and input size

        :param imgs: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param n_classes: int with number of classes in seg output.
        """
        side_x, side_y = self.input_size
        n_classes = self.n_classes
        n_samples, c, h, w = imgs.shape
        n_x = int(h / float(side_x) + 1)
        n_y = int(w / float(side_y) + 1)
        stride_x = (h - side_x) / float(n_x)
        stride_y = (w - side_y) / float(n_y)
        x_ends = [[int(i * stride_x), int(i * stride_x) + side_x] for i in range(n_x + 1)]
        y_ends = [[int(i * stride_y), int(i * stride_y) + side_y] for i in range(n_y + 1)]
        pred = np.zeros([n_samples, n_classes, h, w])
        count = np.zeros([h, w])
        slice_count = 0
        for sx, ex in x_ends:
            for sy, ey in y_ends:
                slice_count += 1
                imgs_slice = imgs[:, :, sx:ex, sy:ey]
                if include_flip_mode:
                    imgs_slice_flip = torch.from_numpy(np.copy(imgs_slice.cpu().numpy()[:, :, :, ::-1])).float()
                is_model_on_cuda = next(self.parameters()).is_cuda
                inp = Variable(imgs_slice, volatile=True)
                if include_flip_mode:
                    flp = Variable(imgs_slice_flip, volatile=True)
                if is_model_on_cuda:
                    inp = inp
                    if include_flip_mode:
                        flp = flp
                psub1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1
                pred[:, :, sx:ex, sy:ey] = psub
                count[sx:ex, sy:ey] += 1.0
        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


class linknetUp(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(linknetUp, self).__init__()
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters / 2, k_size=1, stride=1, padding=1)
        self.deconvbnrelu2 = nn.deconv2DBatchNormRelu(n_filters / 2, n_filters / 2, k_size=3, stride=2, padding=0)
        self.convbnrelu3 = conv2DBatchNormRelu(n_filters / 2, n_filters, k_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class linknet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(linknet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.layers = [2, 2, 2, 2]
        filters = [64, 128, 256, 512]
        filters = [(x / self.feature_scale) for x in filters]
        self.inplanes = filters[0]
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels=3, k_size=7, n_filters=64, padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = residualBlock
        self.encoder1 = self._make_layer(block, filters[0], self.layers[0])
        self.encoder2 = self._make_layer(block, filters[1], self.layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], self.layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.decoder4 = linknetUp(filters[3], filters[2])
        self.decoder4 = linknetUp(filters[2], filters[1])
        self.decoder4 = linknetUp(filters[1], filters[0])
        self.decoder4 = linknetUp(filters[0], filters[0])
        self.finaldeconvbnrelu1 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32 / feature_scale, 3, 2, 1), nn.BatchNorm2d(32 / feature_scale), nn.ReLU(inplace=True))
        self.finalconvbnrelu2 = conv2DBatchNormRelu(in_channels=32 / feature_scale, k_size=3, n_filters=32 / feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv2d(32 / feature_scale, n_classes, 2, 2, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4)
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)
        return f3


pspnet_specs = {'pascal': {'n_classes': 21, 'input_size': (473, 473), 'block_config': [3, 4, 23, 3]}, 'cityscapes': {'n_classes': 19, 'input_size': (713, 713), 'block_config': [3, 4, 23, 3]}, 'ade20k': {'n_classes': 150, 'input_size': (473, 473), 'block_config': [3, 4, 6, 3]}}


class pspnet(nn.Module):
    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(self, n_classes=21, block_config=[3, 4, 23, 3], input_size=(473, 473), version=None):
        super(pspnet, self).__init__()
        self.block_config = pspnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = pspnet_specs[version]['input_size'] if version is not None else input_size
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False)
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)
        self.convbnrelu4_aux = conv2DBatchNormRelu(in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False)
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        inp_shape = x.shape[2:]
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        if self.training:
            x_aux = self.convbnrelu4_aux(x)
            x_aux = self.dropout(x_aux)
            x_aux = self.aux_cls(x_aux)
        x = self.res_block5(x)
        x = self.pyramid_pooling(x)
        x = self.cbr_final(x)
        x = self.dropout(x)
        x = self.classification(x)
        x = F.interpolate(x, size=inp_shape, mode='bilinear', align_corners=True)
        if self.training:
            return x, x_aux
        else:
            return x

    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData']

        def _get_layer_params(layer, ltype):
            if ltype == 'BNData':
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]
            elif ltype in ['ConvolutionData', 'HoleConvolutionData']:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]
            elif ltype == 'InnerProduct':
                raise Exception('Fully connected layers {}, not supported'.format(ltype))
            else:
                raise Exception('Unkown layer type {}'.format(ltype))
        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())
        layer_types = {}
        layer_params = {}
        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                None
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False
            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())
            None
            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))
            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                None
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]
            _transfer_conv(conv_layer_name, conv_module)
            mean, var, gamma, beta = layer_params[conv_layer_name + '/bn']
            None
            bn_module.running_mean.copy_(torch.from_numpy(mean).view_as(bn_module.running_mean))
            bn_module.running_var.copy_(torch.from_numpy(var).view_as(bn_module.running_var))
            bn_module.weight.data.copy_(torch.from_numpy(gamma).view_as(bn_module.weight))
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]
            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {(prefix + '_1_1x1_reduce'): bottleneck.cbr1.cbr_unit, (prefix + '_1_3x3'): bottleneck.cbr2.cbr_unit, (prefix + '_1_1x1_proj'): bottleneck.cb4.cb_unit, (prefix + '_1_1x1_increase'): bottleneck.cb3.cb_unit}
            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)
            for layer_idx in range(2, n_layers + 1):
                residual_layer = block_module.layers[layer_idx - 1]
                residual_conv_bn_dic = {'_'.join(map(str, [prefix, layer_idx, '1x1_reduce'])): residual_layer.cbr1.cbr_unit, '_'.join(map(str, [prefix, layer_idx, '3x3'])): residual_layer.cbr2.cbr_unit, '_'.join(map(str, [prefix, layer_idx, '1x1_increase'])): residual_layer.cb3.cb_unit}
                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)
        convbn_layer_mapping = {'conv1_1_3x3_s2': self.convbnrelu1_1.cbr_unit, 'conv1_2_3x3': self.convbnrelu1_2.cbr_unit, 'conv1_3_3x3': self.convbnrelu1_3.cbr_unit, 'conv5_3_pool6_conv': self.pyramid_pooling.paths[0].cbr_unit, 'conv5_3_pool3_conv': self.pyramid_pooling.paths[1].cbr_unit, 'conv5_3_pool2_conv': self.pyramid_pooling.paths[2].cbr_unit, 'conv5_3_pool1_conv': self.pyramid_pooling.paths[3].cbr_unit, 'conv5_4': self.cbr_final.cbr_unit, ('conv4_' + str(self.block_config[2] + 1)): self.convbnrelu4_aux.cbr_unit}
        residual_layers = {'conv2': [self.res_block2, self.block_config[0]], 'conv3': [self.res_block3, self.block_config[1]], 'conv4': [self.res_block4, self.block_config[2]], 'conv5': [self.res_block5, self.block_config[3]]}
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)
        _transfer_conv('conv6', self.classification)
        _transfer_conv('conv6_1', self.aux_cls)
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

    def tile_predict(self, imgs, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.

        Strides are adaptively computed from the imgs shape
        and input size

        :param imgs: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param n_classes: int with number of classes in seg output.
        """
        side_x, side_y = self.input_size
        n_classes = self.n_classes
        n_samples, c, h, w = imgs.shape
        n_x = int(h / float(side_x) + 1)
        n_y = int(w / float(side_y) + 1)
        stride_x = (h - side_x) / float(n_x)
        stride_y = (w - side_y) / float(n_y)
        x_ends = [[int(i * stride_x), int(i * stride_x) + side_x] for i in range(n_x + 1)]
        y_ends = [[int(i * stride_y), int(i * stride_y) + side_y] for i in range(n_y + 1)]
        pred = np.zeros([n_samples, n_classes, h, w])
        count = np.zeros([h, w])
        slice_count = 0
        for sx, ex in x_ends:
            for sy, ey in y_ends:
                slice_count += 1
                imgs_slice = imgs[:, :, sx:ex, sy:ey]
                if include_flip_mode:
                    imgs_slice_flip = torch.from_numpy(np.copy(imgs_slice.cpu().numpy()[:, :, :, ::-1])).float()
                is_model_on_cuda = next(self.parameters()).is_cuda
                inp = Variable(imgs_slice, volatile=True)
                if include_flip_mode:
                    flp = Variable(imgs_slice_flip, volatile=True)
                if is_model_on_cuda:
                    inp = inp
                    if include_flip_mode:
                        flp = flp
                psub1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1
                pred[:, :, sx:ex, sy:ey] = psub
                count[sx:ex, sy:ey] += 1.0
        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


class refinenet(nn.Module):
    """
    RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
    URL: https://arxiv.org/abs/1611.06612

    References:
    1) Original Author's MATLAB code: https://github.com/guosheng/refinenet
    2) TF implementation by @eragonruan: https://github.com/eragonruan/refinenet-image-segmentation
    """

    def __init__(self, n_classes=21):
        super(refinenet, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        pass


class segnetDown2(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):

    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class segnet(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        return up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        features = list(vgg16.features.children())
        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)
        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit, conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
        assert len(vgg_layers) == len(merged_layers)
        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0), nn.BatchNorm2d(out_size), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0), nn.BatchNorm2d(out_size), nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        return final


class deconv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()
        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias), nn.BatchNorm2d(int(n_filters)))

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()
        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias), nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels, n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters, n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters, n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class residualConvUnit(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(residualConvUnit, self).__init__()
        self.residual_conv_unit = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(channels, channels, kernel_size=kernel_size), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, kernel_size=kernel_size))

    def forward(self, x):
        input = x
        x = self.residual_conv_unit(x)
        return x + input


class multiResolutionFusion(nn.Module):

    def __init__(self, channels, up_scale_high, up_scale_low, high_shape, low_shape):
        super(multiResolutionFusion, self).__init__()
        self.up_scale_high = up_scale_high
        self.up_scale_low = up_scale_low
        self.conv_high = nn.Conv2d(high_shape[1], channels, kernel_size=3)
        if low_shape is not None:
            self.conv_low = nn.Conv2d(low_shape[1], channels, kernel_size=3)

    def forward(self, x_high, x_low):
        high_upsampled = F.upsample(self.conv_high(x_high), scale_factor=self.up_scale_high, mode='bilinear')
        if x_low is None:
            return high_upsampled
        low_upsampled = F.upsample(self.conv_low(x_low), scale_factor=self.up_scale_low, mode='bilinear')
        return low_upsampled + high_upsampled


class chainedResidualPooling(nn.Module):

    def __init__(self, channels, input_shape):
        super(chainedResidualPooling, self).__init__()
        self.chained_residual_pooling = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(5, 1, 2), nn.Conv2d(input_shape[1], channels, kernel_size=3))

    def forward(self, x):
        input = x
        x = self.chained_residual_pooling(x)
        return x + input


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RU,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (bottleNeckIdentifyPSP,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (bottleNeckPSP,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (cascadeFeatureFusion,
     lambda: ([], {'n_classes': 4, 'low_in_channels': 4, 'high_in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 7, 7]), torch.rand([4, 4, 13, 13])], {}),
     False),
    (conv2DBatchNorm,
     lambda: ([], {'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv2DBatchNormRelu,
     lambda: ([], {'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (deconv2DBatchNorm,
     lambda: ([], {'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (deconv2DBatchNormRelu,
     lambda: ([], {'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (fcn16s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (fcn32s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (fcn8s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (frrn,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (multiResolutionFusion,
     lambda: ([], {'channels': 4, 'up_scale_high': 1.0, 'up_scale_low': 1.0, 'high_shape': [4, 4], 'low_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (pspnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (pyramidPooling,
     lambda: ([], {'in_channels': 4, 'pool_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (refinenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (residualBlock,
     lambda: ([], {'in_channels': 4, 'n_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (residualBlockPSP,
     lambda: ([], {'n_blocks': 4, 'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (segnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (segnetDown2,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (segnetDown3,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (unetConv2,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'is_batchnorm': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_meetshah1995_pytorch_semseg(_paritybench_base):
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

