import sys
_module = sys.modules[__name__]
del sys
dataset = _module
voc_aug_test = _module
voc_aug_train = _module
models = _module
context_pooling = _module
deeplab = _module
deeplab3 = _module
erfnet = _module
fc_resnet = _module
fc_sense_resnet = _module
fcn = _module
pspnet = _module
sync_bn = _module
sync_bn_lib = _module
build = _module
functions = _module
sync_bn = _module
modules = _module
sync_bn = _module
options = _module
road_npy2img = _module
test_erfnet_multi_scale = _module
test_pspnet_multi_scale = _module
trainId2color = _module
train_erfnet_vanilla = _module
train_pspnet = _module
transforms = _module
transforms_train = _module

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


import torch


from torch.utils.data import Dataset


import torch.nn as nn


from torch import nn


import torch.nn.init as init


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


import torch.cuda.nccl as nccl


from torch.autograd import Function


from torch.autograd import Variable


import time


import torchvision


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import random


import numbers


class ASSP(nn.Module):

    def __init__(self, in_channels, channels, kernel_size=3, dilation_series=[6, 12, 18, 24]):
        super(ASSP, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.feature_dim = channels
        for dilation in dilation_series:
            padding = dilation * int((kernel_size - 1) / 2)
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(1, len(self.conv2d_list)):
            out += self.conv2d_list[i](x)
        return out


class ASSP3(nn.Module):

    def __init__(self, in_channels, channels=256, kernel_size=3, dilation_series=[6, 12, 18]):
        super(ASSP3, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.bn2d_list = nn.ModuleList()
        self.feature_dim = channels * (len(dilation_series) + 1) + in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False))
        self.bn2d_list.append(nn.BatchNorm2d(channels))
        for dilation in dilation_series:
            padding = dilation * int((kernel_size - 1) / 2)
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False))
            self.bn2d_list.append(nn.BatchNorm2d(channels))
        self.out_conv2d = nn.Conv2d(self.feature_dim, channels, kernel_size=1, stride=1, bias=False)
        self.out_bn2d = nn.BatchNorm2d(channels)
        self.feature_dim = channels

    def forward(self, x):
        outs = []
        input_size = tuple(x.size()[2:4])
        for i in range(len(self.conv2d_list)):
            outs.append(self.conv2d_list[i](x))
            outs[i] = self.bn2d_list[i](outs[i])
            outs[i] = self.relu(outs[i])
        outs.append(nn.functional.upsample(nn.functional.avg_pool2d(x, input_size), size=input_size))
        out = torch.cat(tuple(outs), dim=1)
        out = self.out_conv2d(out)
        out = self.out_bn2d(out)
        out = self.relu(out)
        return out


class PSPP(nn.Module):

    def __init__(self, in_channels, channels=512, scale_series=[10, 20, 30, 60]):
        super(PSPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_list = nn.ModuleList()
        self.bn2d_list = nn.ModuleList()
        self.scale_series = scale_series[:]
        self.feature_dim = channels * len(scale_series) + in_channels
        for i in range(len(scale_series)):
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False))
            self.bn2d_list.append(nn.BatchNorm2d(channels))
        self.out_conv2d = nn.Conv2d(self.feature_dim, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_bn2d = nn.BatchNorm2d(channels)
        self.feature_dim = channels

    def forward(self, x):
        outs = []
        input_size = tuple(x.size()[2:4])
        outs.append(x)
        for i in range(len(self.scale_series)):
            shrink_size = max(1, int((input_size[0] - 1) / self.scale_series[i] + 1)), max(1, int((input_size[1] - 1) / self.scale_series[i] + 1))
            pad_h = shrink_size[0] * self.scale_series[i] - input_size[0]
            pad_w = shrink_size[1] * self.scale_series[i] - input_size[1]
            pad_hw = pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2
            if sum(pad_hw) > 0:
                outs.append(nn.functional.upsample(torch.nn.functional.pad(x, pad_hw, mode='constant', value=0), size=shrink_size, mode='bilinear'))
            else:
                outs.append(nn.functional.upsample(x, size=shrink_size, mode='bilinear'))
            outs[i + 1] = self.conv2d_list[i](outs[i + 1])
            outs[i + 1] = self.bn2d_list[i](outs[i + 1])
            outs[i + 1] = self.relu(outs[i + 1])
            outs[i + 1] = nn.functional.upsample(outs[i + 1], scale_factor=self.scale_series[i], mode='bilinear')
            outs[i + 1] = outs[i + 1][:, :, pad_h // 2:pad_h // 2 + input_size[0], pad_w // 2:pad_w // 2 + input_size[1]]
        out = torch.cat(tuple(outs), dim=1)
        out = self.out_conv2d(out)
        out = self.out_bn2d(out)
        out = self.relu(out)
        return out


class DeepLab(nn.Module):

    def __init__(self, num_class, base_model='resnet101', dropout=0.1, partial_bn=True):
        super(DeepLab, self).__init__()
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self._enable_pbn = partial_bn
        self.num_class = num_class
        if partial_bn:
            self.partialBN(True)
        self._prepare_base_model(base_model)
        self.context_model = context_pooling.ASSP(self.base_model.feature_dim, num_class)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(fc_resnet, 'fc_' + base_model)(pretrained=True)
            self.input_mean = self.base_model.input_mean
            self.input_std = self.base_model.input_std
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(DeepLab, self).train(mode)
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []
        addtional_weight = []
        addtional_bias = []
        addtional_bn = []
        for m in self.base_model.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        if self.context_model is not None:
            for m in self.context_model.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional weight'}, {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 0, 'name': 'addtional bias'}, {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional BN scale/shift'}, {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base weight'}, {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base bias'}, {'params': base_bn, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base BN scale/shift'}]

    def forward(self, x):
        input_size = tuple(x.size()[2:4])
        x, _ = self.base_model(x)
        x = self.dropout(x)
        x = self.context_model(x)
        x = nn.functional.upsample(x, size=input_size, mode='bilinear')
        return x


class DeepLab3(nn.Module):

    def __init__(self, num_class, base_model='resnet101', dropout=0.1, partial_bn=True):
        super(DeepLab3, self).__init__()
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self._enable_pbn = partial_bn
        self.num_class = num_class
        if partial_bn:
            self.partialBN(True)
        self._prepare_base_model(base_model)
        self.context_model = context_pooling.ASSP3(self.base_model.feature_dim)
        self.classifier = nn.Conv2d(self.context_model.feature_dim, num_class, kernel_size=1)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(fc_resnet, 'fc_' + base_model)(pretrained=True)
            self.input_mean = self.base_model.input_mean
            self.input_std = self.base_model.input_std
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(DeepLab3, self).train(mode)
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []
        addtional_weight = []
        addtional_bias = []
        addtional_bn = []
        for m in self.base_model.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        if self.context_model is not None:
            for m in self.context_model.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        if self.classifier is not None:
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional weight'}, {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 0, 'name': 'addtional bias'}, {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional BN scale/shift'}, {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base weight'}, {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base bias'}, {'params': base_bn, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base BN scale/shift'}]

    def forward(self, x):
        input_size = tuple(x.size()[2:4])
        x, _ = self.base_model(x)
        x = self.context_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = nn.functional.upsample(x, size=input_size, mode='bilinear')
        return x


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


class ERFNet(nn.Module):

    def __init__(self, num_classes, partial_bn=False, encoder=None):
        super().__init__()
        if encoder == None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.input_mean = [103.939, 116.779, 123.68]
        self.input_std = [1, 1, 1]
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ERFNet, self).train(mode)
        if self._enable_pbn:
            None
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []
        addtional_weight = []
        addtional_bias = []
        addtional_bn = []
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional weight'}, {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 1, 'name': 'addtional bias'}, {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 0, 'name': 'addtional BN scale/shift'}, {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base weight'}, {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base bias'}, {'params': base_bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'base BN scale/shift'}]

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FCResNet(nn.Module):

    def __init__(self, block, layers):
        super(FCResNet, self).__init__()
        self.input_mean = [103.939, 116.779, 123.68]
        self.input_std = [1, 1, 1]
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        """self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(128, momentum=0.95)"""
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.mid_feature_dim = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.feature_dim = self.inplanes
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, mg=None):
        downsample = None
        if mg is None:
            mg = [(1) for _ in range(blocks)]
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=0.95))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation * mg[0], downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation * mg[i]))
        return nn.Sequential(*layers)

    def forward(self, x, mid_feature=False):
        """x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if mid_feature:
            y = self.layer4(x)
            return y, x
        else:
            x = self.layer4(x)
            return x

    def load_state_dict(self, state_dict):
        state_dict.popitem()
        state_dict.popitem()
        super(FCResNet, self).load_state_dict(state_dict, strict=False)


class FCN(nn.Module):

    def __init__(self, num_class, base_model='resnet50', dropout=0.1, partial_bn=True):
        super(FCN, self).__init__()
        self._enable_pbn = partial_bn
        self.num_class = num_class
        if partial_bn:
            self.partialBN(True)
        self._prepare_base_model(base_model)
        self.classifier = nn.Sequential(nn.Conv2d(self.base_model.feature_dim, 512, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout), nn.Conv2d(512, num_class, kernel_size=1))

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(fc_resnet, 'fc_' + base_model)(pretrained=True)
            self.input_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            self.input_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(FCN, self).train(mode)
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []
        addtional_weight = []
        addtional_bias = []
        addtional_bn = []
        for m in self.base_model.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        if self.classifier is not None:
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional weight'}, {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 0, 'name': 'addtional bias'}, {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional BN scale/shift'}, {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base weight'}, {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base bias'}, {'params': base_bn, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base BN scale/shift'}]

    def forward(self, x):
        input_size = tuple(x.size()[2:4])
        x = self.base_model(x)
        x = self.classifier(x)
        x = nn.functional.upsample(x, size=input_size, mode='bilinear')
        return x


class PSPNet(nn.Module):

    def __init__(self, num_class, base_model='resnet101', dropout=0.1, partial_bn=False, scale_series=[10, 20, 30, 60]):
        super(PSPNet, self).__init__()
        self.dropout = dropout
        self._enable_pbn = partial_bn
        self.num_class = num_class
        if partial_bn:
            self.partialBN(True)
        self._prepare_base_model(base_model)
        self.context_model = context_pooling.PSPP(self.base_model.feature_dim, scale_series=scale_series)
        self.classifier = nn.Conv2d(self.context_model.feature_dim, num_class, kernel_size=1)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(fc_resnet, 'fc_' + base_model)(pretrained=True)
            self.input_mean = self.base_model.input_mean
            self.input_std = self.base_model.input_std
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_aux_loss(self, num_class):
        layers = []
        shrink_dim = int(self.base_model.mid_feature_dim / 4)
        layers.append(nn.Conv2d(self.base_model.mid_feature_dim, shrink_dim, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(shrink_dim))
        layers.append(nn.ReLU(inplace=True))
        if self.dropout > 0:
            layers.append(nn.Dropout2d(p=self.dropout, inplace=True))
        layers.append(nn.Conv2d(shrink_dim, num_class, kernel_size=1))
        self.aux_loss = nn.Sequential(*layers)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(PSPNet, self).train(mode)
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []
        addtional_weight = []
        addtional_bias = []
        addtional_bn = []
        for m in self.base_model.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))
        if self.context_model is not None:
            for m in self.context_model.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        if self.classifier is not None:
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))
        """if self.aux_loss is not None:
            for m in self.aux_loss.modules():
                if isinstance(m, nn.Conv2d):
                    ps = list(m.parameters())
                    addtional_weight.append(ps[0])
                    if len(ps) == 2:
                        addtional_bias.append(ps[1])
                elif isinstance(m, nn.BatchNorm2d):
                    addtional_bn.extend(list(m.parameters()))"""
        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional weight'}, {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 1, 'name': 'addtional bias'}, {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 0, 'name': 'addtional BN scale/shift'}, {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base weight'}, {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base bias'}, {'params': base_bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'base BN scale/shift'}]

    def forward(self, x):
        input_size = tuple(x.size()[2:4])
        x = self.base_model(x)
        x = self.context_model(x)
        x = nn.functional.dropout2d(x, p=self.dropout, training=self.training, inplace=False)
        x = self.classifier(x)
        x = nn.functional.upsample(x, size=input_size, mode='bilinear')
        return x


class Synchronize:
    has_Listener = False
    device_num = 1
    data_ready = []
    data_list = []
    result_list = []
    result_ready = []

    def init(device_num):
        if Synchronize.has_Listener:
            return
        else:
            Synchronize.has_Listener = True
        Synchronize.device_num = device_num
        Synchronize.data_list = [None] * device_num
        Synchronize.result_list = [None] * device_num
        Synchronize.data_ready = [threading.Event() for _ in range(device_num)]
        Synchronize.result_ready = [threading.Event() for _ in range(device_num)]
        for i in range(Synchronize.device_num):
            Synchronize.data_ready[i].clear()
            Synchronize.result_ready[i].set()

        def _worker():
            while True:
                for i in range(Synchronize.device_num):
                    Synchronize.data_ready[i].wait()
                total_sum = Synchronize.data_list[0].cpu().clone()
                for i in range(1, Synchronize.device_num):
                    total_sum = total_sum + Synchronize.data_list[i].cpu()
                for i in range(0, Synchronize.device_num):
                    with torch.cuda.device_of(Synchronize.data_list[i]):
                        Synchronize.result_list[i] = total_sum.clone()
                for i in range(Synchronize.device_num):
                    Synchronize.data_ready[i].clear()
                    Synchronize.result_ready[i].set()
        thread = threading.Thread(target=_worker)
        thread.daemon = True
        thread.start()

    def all_reduce_thread(input):
        if not Synchronize.has_Listener:
            return input
        input_device = input.get_device()
        Synchronize.data_list[input_device] = input
        with torch.device(input_device):
            Synchronize.result_list[input_device] = type(input)(input.size()).zero_()
        Synchronize.result_ready[input_device].clear()
        Synchronize.data_ready[input_device].set()
        Synchronize.result_ready[input_device].wait()
        return Synchronize.result_list[input_device]

    def forward(ctx, input):
        return Synchronize.all_reduce_thread(input)

    def backward(ctx, gradOutput):
        return Synchronize.all_reduce_thread(gradOutput)


class _sync_batch_norm(Function):

    def __init__(self, momentum, eps):
        super(_sync_batch_norm, self).__init__()
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, running_mean, running_var, weight, bias):
        allreduce_num = Synchronize.device_num
        with torch.cuda.device_of(input):
            mean = input.new().resize_(input.size(1)).zero_()
            var = input.new().resize_(input.size(1)).zero_()
            x_std = input.new().resize_(input.size(1)).zero_()
            x_norm = input.new().resize_as_(input)
            output = input.new().resize_as_(input)
        sync_bn_lib.bn_forward_mean_before_allreduce(input, mean, allreduce_num)
        mean = Synchronize.all_reduce_thread(mean)
        sync_bn_lib.bn_forward_var_before_allreduce(input, mean, var, output, allreduce_num)
        var = Synchronize.all_reduce_thread(var)
        sync_bn_lib.bn_forward_after_allreduce(mean, running_mean, var, running_var, x_norm, x_std, weight, bias, output, self.eps, 1.0 - self.momentum)
        self.save_for_backward(weight, bias)
        self.mean = mean
        self.x_norm = x_norm
        self.x_std = x_std
        return output

    def backward(self, grad_output):
        weight, bias = self.saved_tensors
        allreduce_num = Synchronize.device_num
        with torch.cuda.device_of(grad_output):
            grad_input = grad_output.new().resize_as_(grad_output).zero_()
            grad_weight = grad_output.new().resize_as_(weight).zero_()
            grad_bias = grad_output.new().resize_as_(bias).zero_()
            grad_local_weight = grad_output.new().resize_as_(weight).zero_()
            grad_local_bias = grad_output.new().resize_as_(bias).zero_()
        sync_bn_lib.bn_backward_before_allreduce(grad_output, self.x_norm, self.mean, self.x_std, grad_input, grad_local_weight, grad_local_bias, grad_weight, grad_bias)
        grad_local_weight = Synchronize.all_reduce_thread(grad_local_weight)
        grad_local_bias = Synchronize.all_reduce_thread(grad_local_bias)
        sync_bn_lib.bn_backward_after_allreduce(grad_output, self.x_norm, grad_local_weight, grad_local_bias, weight, self.x_std, grad_input, allreduce_num)
        return grad_input, None, None, grad_weight, grad_bias


def sync_batch_norm(input, running_mean, running_var, weight=None, bias=None, momentum=0.1, eps=1e-05):
    """Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _torch_ext.batchnormtrain:

    .. math::

        y = \\frac{x - \\mu[x]}{ \\sqrt{var[x] + \\epsilon}} * \\gamma + \\beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _sync_batch_norm(momentum, eps)(input, running_mean, running_var, weight, bias)


class SyncBatchNorm2d(torch.nn.BatchNorm2d):
    """Synchronized Batch Normalization 2d
    Please use compatible :class:`torch_ext.parallel.SelfDataParallel` and :class:`torch_ext.nn`

    Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - \\mu[x]}{ \\sqrt{var[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable
            affine parameters. Default: True

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> m = torch_ext.nn.BatchNorm2d(100).cuda()
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45)).cuda()
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training and Synchronize.device_num > 1:
                B, C, H, W = input.size()
                rm = Variable(self.running_mean, requires_grad=False)
                rv = Variable(self.running_var, requires_grad=False)
                output = sync_batch_norm(input.view(B, C, -1).contiguous(), rm, rv, self.weight, self.bias, self.momentum, self.eps)
                self.running_mean = rm.data
                self.running_var = rv.data
                return output.view(B, C, H, W)
            else:
                return super(SyncBatchNorm2d, self).forward(input)
        else:
            raise RuntimeError('unknown input type')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASSP,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASSP3,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (ERFNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Encoder,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PSPP,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SyncBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsamplerBlock,
     lambda: ([], {'ninput': 4, 'noutput': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (non_bottleneck_1d,
     lambda: ([], {'chann': 4, 'dropprob': 0.5, 'dilated': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cardwing_Codes_for_IntRA_KD(_paritybench_base):
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

