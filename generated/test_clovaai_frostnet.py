import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
latency_check = _module
models = _module
cifar = _module
alexnet = _module
mobilenet = _module
resnet = _module
utils = _module
mobilenet = _module
mobilenetv3 = _module
resnet = _module
utils = _module
vgg = _module
densenet = _module
inception = _module
mnasnet = _module
mobilenet = _module
resnet = _module
squeezenet = _module
utils = _module
imagenet = _module
alexnet = _module
mobilenet = _module
mobilenetv3 = _module
resnet = _module
shufflenetv2 = _module
utils = _module
vgg = _module
train = _module
Criteria = _module
Tensor_logger = _module
data_functions = _module
flops_compute = _module
flops_counter = _module
helper_functions = _module
optimizer = _module
utils = _module
data = _module
coco = _module
config = _module
voc0712 = _module
layers = _module
box_utils = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
qeval_convert = _module
qtdsod = _module
qtrainval = _module
ssd_qmv2 = _module
augmentations = _module
optimizer = _module
general_details = _module
generate_mappings = _module
print_utils = _module
process_cityscapes = _module
cityscapes = _module
coco = _module
custom_dataset_loader = _module
voc = _module
evaluate = _module
latency_check = _module
multi_box_loss = _module
segmentation_loss = _module
backbones = _module
base = _module
espnet = _module
espnetv2 = _module
mobilenetv2 = _module
mobilenetv3 = _module
espnet = _module
espnetv2 = _module
LRASPP = _module
RASPP = _module
basic = _module
espnet_utils = _module
mobilenetv2 = _module
mobilenetv3 = _module
train = _module
box_utils = _module
color_map = _module
data_transforms = _module
flops_compute = _module
lr_scheduler = _module
classification_accuracy = _module
evaluate_detection = _module
segmentation_miou = _module
voc_helper = _module
nms = _module
optimizer = _module
parallel_wrapper = _module
train_eval_seg = _module
utils = _module
data = _module
aligned_dataset = _module
base_dataset = _module
colorization_dataset = _module
image_folder = _module
single_dataset = _module
template_dataset = _module
unaligned_dataset = _module
combine_A_and_B = _module
make_dataset_aligned = _module
prepare_cityscapes_dataset = _module
base_model = _module
colorization_model = _module
cycle_gan_model = _module
networks = _module
pix2pix_model = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
batch_hed = _module
util = _module
test_before_push = _module
test = _module
train = _module
get_data = _module
html = _module
image_pool = _module
optimizer = _module
util = _module
visualizer = _module
frostnet = _module
frostnet_features = _module
optimizer = _module

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


import time


import numpy as np


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.quantization


from torchvision.transforms import functional as F


from torch.quantization import QuantStub


from torch.quantization import DeQuantStub


from torch.quantization import fuse_modules


from torch._jit_internal import Optional


from torch import nn


import torch.nn.functional as F


import re


import torch.utils.checkpoint as cp


from collections import OrderedDict


from torch import Tensor


from torch.jit.annotations import List


from collections import namedtuple


import warnings


from torch.jit.annotations import Optional


import math


import torch.nn.init as init


from torchvision.models.mobilenet import MobileNetV2


import torchvision.models.shufflenetv2


import random


from torch.quantization.fake_quantize import FakeQuantize


import torchvision


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.autograd import Variable


from functools import reduce


import torch.utils.data as data


from torch.autograd import Function


import torchvision.ops as ops


from math import sqrt as sqrt


from itertools import product as product


from torchvision import transforms


import types


from numpy import random


from torch.utils import data


from torch.nn import functional as F


from torch.nn import init


import numbers


from torchvision.transforms import Pad


from torch.cuda._utils import _get_device_index


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel import gather


import torch.utils.data


from abc import ABC


from abc import abstractmethod


import itertools


import functools


from torch.optim import lr_scheduler


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=False), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=False), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        fuse_modules(self.features, [['0', '1'], ['3', '4'], ['6', '7'], ['8', '9'], ['10', '11']], inplace=True)
        fuse_modules(self.classifier, [['1', '2'], ['4', '5']], inplace=True)


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(False))

    def forward(self, x):
        x = self.conv(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1', '2'], inplace=True)


class _ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.cb = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.cb(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.relu6 = relu6
        self.cbr = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(True) if relu6 else nn.ReLU(False))

    def forward(self, x):
        x = self.cbr(x)
        return x

    def fuse_model(self):
        if self.relu6:
            torch.quantization.fuse_modules(self.cbr, ['0', '1'], inplace=True)
        else:
            torch.quantization.fuse_modules(self.cbr, ['0', '1', '2'], inplace=True)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([_ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation, groups=inter_channels, relu6=True, norm_layer=norm_layer), _ConvBN(inter_channels, out_channels, 1)])
        self.conv = nn.Sequential(*layers)
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()
        if self.expand_ratio != 1:
            self.conv[2].fuse_model()


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNetV2, self).__init__()
        layer1_setting = [[1, 16, 1, 1]]
        layer2_setting = [[6, 24, 2, 2]]
        layer3_setting = [[6, 32, 3, 2]]
        layer4_setting = [[6, 64, 4, 2], [6, 96, 3, 1]]
        if dilated:
            layer5_setting = [[6, 160, 3, 2], [6, 320 // 2, 1, 1]]
        else:
            layer5_setting = [[6, 160, 3, 2], [6, 320, 1, 1]]
        self.in_channels = int(32 * width_mult) if width_mult > 1.0 else 32
        last_channels = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv1 = _ConvBNReLU(3, self.in_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)
        self.layer1 = self._make_layer(InvertedResidual, layer1_setting, width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(InvertedResidual, layer2_setting, width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(InvertedResidual, layer3_setting, width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult, dilation=2, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult, norm_layer=norm_layer)
        if not dilated:
            self.classifier = nn.Sequential(_ConvBNReLU(self.in_channels, last_channels, 1, relu6=True, norm_layer=norm_layer), nn.AdaptiveAvgPool2d(1), nn.Dropout2d(0.2), nn.Conv2d(last_channels, num_classes, 1))
        self.dilated = dilated
        self._init_weight()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for t, c, n, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if dilation == 1 else 1
            layers.append(block(self.in_channels, out_channels, stride, t, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels, out_channels, 1, t, 1, norm_layer=norm_layer))
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c1 = x
        x = self.layer3(x)
        c2 = x
        x = self.layer4(x)
        c3 = x
        x = self.layer5(x)
        c4 = x
        if self.dilated:
            return c1, c2, c3, c4
        else:
            x = self.classifier(x)
            x = x.view(x.size(0), -1)
            return x

    def fuse_model(self):
        self.conv1.fuse_model()
        for layer in self.layer1:
            layer.fuse_model()
        for layer in self.layer2:
            layer.fuse_model()
        for layer in self.layer3:
            layer.fuse_model()
        for layer in self.layer4:
            layer.fuse_model()
        for layer in self.layer5:
            layer.fuse_model()
        if not self.dilated:
            self.classifier[0].fuse_model()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Identity(nn.Module):

    def forward(self, x):
        return x


class _Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(_Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
        self.quant_add = nn.quantized.FloatFunctional()
        self.quant_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.quant_add.add_scalar(x, 3.0)
        out = self.relu6(out)
        out = self.quant_mul.mul_scalar(out, 1 / 6)
        return out


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=False), nn.ReLU(inplace=False), nn.Linear(in_channels // reduction, in_channels, bias=False), _Hsigmoid(True))
        self.quant_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return self.quant_mul.mul(x, out.expand_as(x))

    def fuse_model(self):
        torch.quantization.fuse_modules(self.fc, ['0', '1'], inplace=True)


class _Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.sigmoid = _Hsigmoid()
        self.quant_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.sigmoid(x)
        out = self.quant_mul.mul(x, out)
        return out


class _ConvBNHswish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.cb = _ConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.cb(x)
        x = self.act(x)
        return x

    def fuse_model(self):
        self.cb.fuse_model()


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=False, nl='RE', norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity
        self.conv = nn.Sequential(_ConvBNHswish(in_channels, exp_size, 1) if nl == 'HS' else _ConvBNReLU(in_channels, exp_size, 1), _ConvBNHswish(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation, dilation, groups=exp_size) if nl == 'HS' else _ConvBNReLU(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation, dilation, groups=exp_size), SELayer(exp_size), _ConvBN(exp_size, out_channels, 1))
        self.se = se
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()
        if self.se:
            self.conv[2].fuse_model()
        self.conv[3].fuse_model()


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
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
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


shufflenetv2 = sys.modules['torchvision.models.shufflenetv2']


class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):

    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.cat.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = self.cat.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shufflenetv2.channel_shuffle(out, 2)
        return out


class QuantizableMobileNetV2(MobileNetV2):

    def __init__(self, *args, **kwargs):
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()


class MobileNetV3(nn.Module):

    def __init__(self, nclass=1000, mode='large', width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, RE=False, **kwargs):
        super(MobileNetV3, self).__init__()
        if RE:
            if mode == 'large':
                layer1_setting = [[3, 16, 16, False, 'RE', 1], [3, 64, 24, False, 'RE', 2], [3, 72, 24, False, 'RE', 1]]
                layer2_setting = [[5, 72, 40, True, 'RE', 2], [5, 120, 40, True, 'RE', 1], [5, 120, 40, True, 'RE', 1]]
                layer3_setting = [[3, 240, 80, False, 'RE', 2], [3, 200, 80, False, 'RE', 1], [3, 184, 80, False, 'RE', 1], [3, 184, 80, False, 'RE', 1], [3, 480, 112, True, 'RE', 1], [3, 672, 112, True, 'RE', 1]]
                if dilated:
                    layer4_setting = [[5, 672, 160, True, 'RE', 2], [5, 960, 160, True, 'RE', 1], [5, 960 // 2, 160 // 2, True, 'RE', 1]]
                else:
                    layer4_setting = [[5, 672, 160, True, 'RE', 2], [5, 960, 160, True, 'RE', 1], [5, 960, 160, True, 'RE', 1]]
            elif mode == 'small':
                layer1_setting = [[3, 16, 16, True, 'RE', 2]]
                layer2_setting = [[3, 72, 24, False, 'RE', 2], [3, 88, 24, False, 'RE', 1]]
                layer3_setting = [[5, 96, 40, True, 'RE', 2], [5, 240, 40, True, 'RE', 1], [5, 240, 40, True, 'RE', 1], [5, 120, 48, True, 'RE', 1], [5, 144, 48, True, 'RE', 1]]
                if dilated:
                    layer4_setting = [[5, 288, 96, True, 'RE', 2], [5, 576, 96, True, 'RE', 1], [5, 576 // 2, 96 // 2, True, 'RE', 1]]
                else:
                    layer4_setting = [[5, 288, 96, True, 'RE', 2], [5, 576, 96, True, 'RE', 1], [5, 576, 96, True, 'RE', 1]]
            else:
                raise ValueError('Unknown mode.')
        elif mode == 'large':
            layer1_setting = [[3, 16, 16, False, 'RE', 1], [3, 64, 24, False, 'RE', 2], [3, 72, 24, False, 'RE', 1]]
            layer2_setting = [[5, 72, 40, True, 'RE', 2], [5, 120, 40, True, 'RE', 1], [5, 120, 40, True, 'RE', 1]]
            layer3_setting = [[3, 240, 80, False, 'HS', 2], [3, 200, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 480, 112, True, 'HS', 1], [3, 672, 112, True, 'HS', 1]]
            if dilated:
                layer4_setting = [[5, 672, 160, True, 'HS', 2], [5, 960, 160, True, 'HS', 1], [5, 960 // 2, 160 // 2, True, 'HS', 1]]
            else:
                layer4_setting = [[5, 672, 160, True, 'HS', 2], [5, 960, 160, True, 'HS', 1], [5, 960, 160, True, 'HS', 1]]
        elif mode == 'small':
            layer1_setting = [[3, 16, 16, True, 'RE', 2]]
            layer2_setting = [[3, 72, 24, False, 'RE', 2], [3, 88, 24, False, 'RE', 1]]
            layer3_setting = [[5, 96, 40, True, 'HS', 2], [5, 240, 40, True, 'HS', 1], [5, 240, 40, True, 'HS', 1], [5, 120, 48, True, 'HS', 1], [5, 144, 48, True, 'HS', 1]]
            if dilated:
                layer4_setting = [[5, 288, 96, True, 'HS', 2], [5, 576, 96, True, 'HS', 1], [5, 576 // 2, 96 // 2, True, 'HS', 1]]
            else:
                layer4_setting = [[5, 288, 96, True, 'HS', 2], [5, 576, 96, True, 'HS', 1], [5, 576, 96, True, 'HS', 1]]
        else:
            raise ValueError('Unknown mode.')
        self.in_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        if RE:
            self.conv1 = _ConvBNReLU(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)
        else:
            self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)
        self.layer1 = self._make_layer(Bottleneck, layer1_setting, width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Bottleneck, layer2_setting, width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Bottleneck, layer3_setting, width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting, width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting, width_mult, norm_layer=norm_layer)
        classifier = list()
        if mode == 'large':
            if dilated:
                last_bneck_channels = int(960 // 2 * width_mult) if width_mult > 1.0 else 960 // 2
            else:
                last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            if RE:
                self.layer5 = _ConvBNReLU(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            else:
                self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            if not dilated:
                classifier.append(nn.AdaptiveAvgPool2d(1))
                classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
                classifier.append(_Hswish(True))
                classifier.append(nn.Conv2d(1280, nclass, 1))
        elif mode == 'small':
            if dilated:
                last_bneck_channels = int(576 // 2 * width_mult) if width_mult > 1.0 else 576 // 2
            else:
                last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            if RE:
                self.layer5 = _ConvBNReLU(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            else:
                self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            if not dilated:
                classifier.append(SEModule(last_bneck_channels))
                classifier.append(nn.AdaptiveAvgPool2d(1))
                classifier.append(nn.Conv2d(last_bneck_channels, 1024, 1))
                classifier.append(_Hswish(True))
                classifier.append(nn.Conv2d(1024, nclass, 1))
        else:
            raise ValueError('Unknown mode.')
        self.mode = mode
        if not dilated:
            self.classifier = nn.Sequential(*classifier)
        self.dilated = dilated
        self._init_weights()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if dilation == 1 else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c1 = x
        x = self.layer3(x)
        c2 = x
        x = self.layer4(x)
        c3 = x
        x = self.layer5(x)
        c4 = x
        if self.dilated:
            return c1, c2, c3, c4
        else:
            x = self.classifier(x)
            x = x.view(x.size(0), x.size(1))
            return x

    def fuse_model(self):
        self.conv1.fuse_model()
        for layer in self.layer1:
            layer.fuse_model()
        for layer in self.layer2:
            layer.fuse_model()
        for layer in self.layer3:
            layer.fuse_model()
        for layer in self.layer4:
            layer.fuse_model()
        self.layer5.fuse_model()
        if not self.dilated:
            if self.mode == 'small':
                self.classifier[0].fuse_model()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class QuantizableBasicBlock(BasicBlock):

    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add_relu.add_relu(out, identity)
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):

    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)
        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()


class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1'], inplace=True)


class ConvReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), nn.ReLU(False))

    def forward(self, x):
        x = self.conv(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1'], inplace=True)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=False), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=False), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        for idx in range(len(self.features)):
            if type(self.features[idx]) == nn.Conv2d:
                if type(self.features[idx + 1]) == nn.BatchNorm2d:
                    fuse_modules(self.features, [str(idx), str(idx + 1), str(idx + 2)], inplace=True)
                else:
                    fuse_modules(self.features, [str(idx), str(idx + 1)], inplace=True)
        fuse_modules(self.classifier, [['0', '1'], ['3', '4']], inplace=True)


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input):

        def closure(*inputs):
            return self.bn_function(*inputs)
        return cp.checkpoint(closure, input)

    @torch.jit._overload_method
    def forward(self, input):
        pass

    @torch.jit._overload_method
    def forward(self, input):
        pass

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


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
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, inception_blocks=None, init_weights=True):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn('Scripted Inception3 always returns Inception3 Tuple')
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


_BN_MOMENTUM = 1 - 0.9997


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM), _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM), _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM), _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM), _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM), _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM), _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM), nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.zeros_(m.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        assert version in [1, 2]
        if version == 1 and not self.alpha == 1.0:
            depths = _get_depths(self.alpha)
            v1_stem = [nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(16, momentum=_BN_MOMENTUM), _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM)]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer
            self._version = 1
            warnings.warn('A new version of MNASNet model has been implemented. Your checkpoint was saved using the previous version. This checkpoint will load and work as before, but you may want to upgrade by training a newer model or transfer learning from an updated ImageNet checkpoint.', UserWarning)
        super(MNASNet, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        elif version == '1_1':
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        else:
            raise ValueError('Unsupported SqueezeNet version {version}:1_0 or 1_1 expected'.format(version=version))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class CrossEntropyLoss2d(nn.Module):
    """
    This file defines a cross entropy loss for 2D images
    """

    def __init__(self, weight=None, ignore=None):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        """
        super().__init__()
        if int(torch.__version__[2]) < 4:
            self.loss = nn.NLLLoss2d(weight, ignore_index=ignore)
        else:
            self.loss = nn.NLLLoss(weight, ignore_index=ignore)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class MultiBoxLoss(nn.Module):

    def __init__(self, neg_pos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class conv_bn(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, inp, oup, stride=1, k_size=3, padding=1, group=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        self.cbr = nn.Sequential(nn.Conv2d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, groups=group, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.cbr(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0', '1', '2'], inplace=True)


class conv_bn_no_relu(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, inp, oup, stride=1, k_size=3, padding=1, group=1):
        super().__init__()
        self.cb = nn.Sequential(nn.Conv2d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, groups=group, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        return self.cb(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)


class dwd_block(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, inp, oup):
        super().__init__()
        self.dwd1 = conv_bn(inp=inp, oup=oup, stride=1, k_size=1, padding=0)
        self.dwd2 = conv_bn(inp=oup, oup=oup, stride=1, k_size=3, padding=1, group=oup)

    def forward(self, x):
        return self.dwd2(self.dwd1(x))

    def fuse_model(self):
        self.dwd1.fuse_model()
        self.dwd2.fuse_model()


class trans_block(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, inp, oup):
        super().__init__()
        self.trn1 = conv_bn(inp=inp, oup=oup, stride=1, k_size=1, padding=0)
        self.trn2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        return self.trn2(self.trn1(x))

    def fuse_model(self):
        self.trn1.fuse_model()


class downsample_0(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, in_channels, out_channels):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        self.dwn1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, ceil_mode=True)
        self.conv1 = conv_bn(inp=in_channels, oup=out_channels, stride=1, k_size=1, padding=0)

    def forward(self, x):
        return self.conv1(self.dwn1(x))

    def fuse_model(self):
        self.conv1.fuse_model()


class downsample_1(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2 = conv_bn_no_relu(inp=in_channels, oup=out_channels, stride=1, k_size=1, padding=0)
        self.conv3 = conv_bn(inp=out_channels, oup=out_channels, stride=2, k_size=3, padding=1, group=out_channels)

    def forward(self, x):
        return self.conv3(self.conv2(x))

    def fuse_model(self):
        self.conv2.fuse_model()
        self.conv3.fuse_model()


class upsample(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv_bn(inp=in_channels, oup=in_channels, stride=1, k_size=3, padding=1, group=in_channels)

    def forward(self, x):
        return self.conv1(x)

    def fuse_model(self):
        self.conv1.fuse_model()


class baseNet(nn.Module):
    """
        This class defines the basenet
    """

    def __init__(self):
        super().__init__()
        self.base1 = conv_bn(inp=3, oup=64, stride=2, k_size=3, padding=1)
        self.base2 = conv_bn(inp=64, oup=64, stride=1, k_size=1, padding=0)
        self.base3 = conv_bn(inp=64, oup=64, stride=1, k_size=3, padding=1, group=64)
        self.base4 = conv_bn(inp=64, oup=128, stride=1, k_size=1, padding=0)
        self.base5 = conv_bn(inp=128, oup=128, stride=1, k_size=3, padding=1, group=128)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        x = self.base4(x)
        x = self.base5(x)
        x = self.max(x)
        return x

    def fuse_model(self):
        self.base1.fuse_model()
        self.base2.fuse_model()
        self.base3.fuse_model()
        self.base4.fuse_model()
        self.base5.fuse_model()


TDSOD_coco = {'num_classes': 201, 'lr_steps': (280000, 360000, 400000), 'max_iter': 400000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [21, 45, 99, 153, 207, 261], 'max_sizes': [45, 99, 153, 207, 261, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'COCO'}


TDSOD_voc = {'num_classes': 21, 'lr_steps': (120000, 150000, 180000), 'max_iter': 180000, 'feature_maps': [38, 19, 10, 5, 3, 2], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 264, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'VOC'}


class QSSD_TDSOD_Feat(nn.Module):

    def __init__(self, size, num_classes):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.cfg = (TDSOD_coco, TDSOD_voc)[num_classes == 21]
        self.num_feat = len(self.cfg['feature_maps'])
        self.quant = QuantStub()
        self.dequant_list = []
        for idx in range(self.num_feat):
            self.dequant_list.append(DeQuantStub())
        self.dequant_list = nn.ModuleList(self.dequant_list)
        self.base = baseNet()
        self.ddb_0 = []
        self.quant_list_0 = []
        inp = 128
        for it in range(4):
            if it == 0:
                self.ddb_0.append(dwd_block(inp=inp, oup=32))
            else:
                inp += 32
                self.ddb_0.append(dwd_block(inp=inp, oup=32))
            self.quant_list_0.append(nn.quantized.FloatFunctional())
        self.quant_list_0 = nn.ModuleList(self.quant_list_0)
        self.ddb_0 = nn.ModuleList(self.ddb_0)
        self.trans_0 = trans_block(inp=256, oup=128)
        self.ddb_1 = []
        self.quant_list_1 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_1.append(dwd_block(inp=inp, oup=48))
            else:
                inp += 48
                self.ddb_1.append(dwd_block(inp=inp, oup=48))
            self.quant_list_1.append(nn.quantized.FloatFunctional())
        self.quant_list_1 = nn.ModuleList(self.quant_list_1)
        self.ddb_1 = nn.ModuleList(self.ddb_1)
        self.trans_1 = trans_block(inp=416, oup=128)
        self.ddb_2 = []
        self.quant_list_2 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_2.append(dwd_block(inp=inp, oup=64))
            else:
                inp += 64
                self.ddb_2.append(dwd_block(inp=inp, oup=64))
            self.quant_list_2.append(nn.quantized.FloatFunctional())
        self.quant_list_2 = nn.ModuleList(self.quant_list_2)
        self.ddb_2 = nn.ModuleList(self.ddb_2)
        self.trans_2 = conv_bn(inp=512, oup=256, stride=1, k_size=1, padding=0)
        self.ddb_3 = []
        self.quant_list_3 = []
        inp = 256
        for it in range(6):
            if it == 0:
                self.ddb_3.append(dwd_block(inp=inp, oup=80))
            else:
                inp += 80
                self.ddb_3.append(dwd_block(inp=inp, oup=80))
            self.quant_list_3.append(nn.quantized.FloatFunctional())
        self.quant_list_3 = nn.ModuleList(self.quant_list_3)
        self.ddb_3 = nn.ModuleList(self.ddb_3)
        self.trans_3 = conv_bn(inp=736, oup=64, stride=1, k_size=1, padding=0)
        self.downfeat_0 = []
        self.downfeat_1 = []
        for it in range(5):
            if it == 1:
                self.downfeat_0.append(downsample_0(in_channels=128 + 64, out_channels=64))
                self.downfeat_1.append(downsample_1(in_channels=128 + 64, out_channels=64))
            else:
                self.downfeat_0.append(downsample_0(in_channels=128, out_channels=64))
                self.downfeat_1.append(downsample_1(in_channels=128, out_channels=64))
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=128))
        self.downfeat_0 = nn.ModuleList(self.downfeat_0)
        self.downfeat_1 = nn.ModuleList(self.downfeat_1)
        self.upfeat = nn.ModuleList(self.upfeat)
        self.qadd1 = nn.quantized.FloatFunctional()
        self.qadd2 = nn.quantized.FloatFunctional()
        self.qadd3 = nn.quantized.FloatFunctional()
        self.qadd4 = nn.quantized.FloatFunctional()
        self.qadd5 = nn.quantized.FloatFunctional()
        self.qcat0 = nn.quantized.FloatFunctional()
        self.qcat1 = nn.quantized.FloatFunctional()
        self.qcat2 = nn.quantized.FloatFunctional()
        self.qcat3 = nn.quantized.FloatFunctional()
        self.qcat4 = nn.quantized.FloatFunctional()
        self.qcat5 = nn.quantized.FloatFunctional()

    def forward(self, x):
        """applies network layers and ops on input image(s) x.

        args:
            x: input image or batch of images. shape: [batch,3,300,300].

        return:
            depending on phase:
            test:
                variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, shape: [batch*num_priors,num_classes]
                    2: localization layers, shape: [batch,num_priors*4]
                    3: priorbox layers, shape: [2,num_priors*4]
        """
        sources = list()
        x = self.quant(x)
        x = self.base(x)
        for idx, blc in enumerate(self.ddb_0):
            x = self.quant_list_0[idx].cat((x, blc(x)), 1)
        x = self.trans_0(x)
        infeat_1 = x
        for idx, blc in enumerate(self.ddb_1):
            x = self.quant_list_1[idx].cat((x, blc(x)), 1)
        x = self.trans_1(x)
        for idx, blc in enumerate(self.ddb_2):
            x = self.quant_list_2[idx].cat((x, blc(x)), 1)
        x = self.trans_2(x)
        for idx, blc in enumerate(self.ddb_3):
            x = self.quant_list_3[idx].cat((x, blc(x)), 1)
        x = self.trans_3(x)
        infeat_2 = x
        infeat_3 = self.qcat0.cat((self.downfeat_0[0](infeat_1), self.downfeat_1[0](infeat_1)), 1)
        sz_x = infeat_3.size()[2]
        sz_y = infeat_3.size()[3]
        s0 = self.qcat1.cat((infeat_3[:, :, :sz_x, :sz_y], infeat_2[:, :, :sz_x, :sz_y]), 1)
        s1 = self.qcat2.cat((self.downfeat_0[1](s0), self.downfeat_1[1](s0)), 1)
        s2 = self.qcat3.cat((self.downfeat_0[2](s1), self.downfeat_1[2](s1)), 1)
        s3 = self.qcat4.cat((self.downfeat_0[3](s2), self.downfeat_1[3](s2)), 1)
        s4 = self.qcat5.cat((self.downfeat_0[4](s3), self.downfeat_1[4](s3)), 1)
        sources.append(s4)
        u1 = self.qadd1.add(self.upfeat[0](F.interpolate(s4, size=(s3.size()[2], s3.size()[3]), mode='bilinear')), s3)
        sources.append(u1)
        u2 = self.qadd2.add(self.upfeat[1](F.interpolate(u1, size=(s2.size()[2], s2.size()[3]), mode='bilinear')), s2)
        sources.append(u2)
        u3 = self.qadd3.add(self.upfeat[2](F.interpolate(u2, size=(s1.size()[2], s1.size()[3]), mode='bilinear')), s1)
        sources.append(u3)
        u4 = self.qadd4.add(self.upfeat[3](F.interpolate(u3, size=(infeat_3.size()[2], infeat_3.size()[3]), mode='bilinear')), infeat_3)
        sources.append(u4)
        u5 = self.qadd5.add(self.upfeat[4](F.interpolate(u4, size=(infeat_1.size()[2], infeat_1.size()[3]), mode='bilinear')), infeat_1)
        sources.append(u5)
        sources = sources[::-1]
        output = []
        for idx, source in enumerate(sources):
            output.append(self.dequant_list[idx](source))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            mdata = torch.load(base_file, map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            None
        else:
            None
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)

    def fuse_model(self):
        self.base.fuse_model()
        for itm in self.ddb_0:
            itm.fuse_model()
        self.trans_0.fuse_model()
        for itm in self.ddb_1:
            itm.fuse_model()
        self.trans_1.fuse_model()
        for itm in self.ddb_2:
            itm.fuse_model()
        self.trans_2.fuse_model()
        for itm in self.ddb_3:
            itm.fuse_model()
        self.trans_3.fuse_model()
        for itm in self.downfeat_0:
            itm.fuse_model()
        for itm in self.downfeat_1:
            itm.fuse_model()
        for itm in self.upfeat:
            itm.fuse_model()


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def detect(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                _, order = torch.sort(scores, 0, True)
                order = order[:self.top_k]
                keep = nms_faster_rcnn(boxes[order] * float(cfg['min_dim']), scores[order], self.nms_thresh)
                output[i, cl, :len(keep)] = torch.cat((scores[order[keep]].unsqueeze(1), boxes[order[keep]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def get_prior(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class QSSD_TDSOD_HEAD(nn.Module):

    def __init__(self, phase='train', num_classes=21, cfg=[4, 6, 6, 6, 4, 4]):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.loc_layers = []
        self.conf_layers = []
        self.cfg = (TDSOD_coco, TDSOD_voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.get_prior()
        self.priors.requires_grad = False
        self.loc_layers += [conv_bn_no_relu(inp=128, oup=cfg[0] * 4, stride=1, k_size=3, padding=1)]
        self.conf_layers += [conv_bn_no_relu(inp=128, oup=cfg[0] * num_classes, stride=1, k_size=3, padding=1)]
        for k in range(1, 6):
            self.loc_layers += [conv_bn_no_relu(inp=128, oup=cfg[k] * 4, stride=1, k_size=3, padding=1)]
            self.conf_layers += [conv_bn_no_relu(inp=128, oup=cfg[k] * num_classes, stride=1, k_size=3, padding=1)]
        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)

    def forward(self, sources):
        loc = []
        conf = []
        for idx, (x, l, c) in enumerate(zip(sources, self.loc_layers, self.conf_layers)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect.detect(loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes)), self.priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
        return output


class SSD_MobileNetV2_Feat(nn.Module):

    def __init__(self, size, base, extras, extras_head_pos, use_final_conv):
        super(SSD_MobileNetV2_Feat, self).__init__()
        self.size = size
        self.quant = QuantStub()
        self.dequant_list = []
        for idx in range(len(extras_head_pos) + 2):
            self.dequant_list.append(DeQuantStub())
        self.dequant_list = nn.ModuleList(self.dequant_list)
        self.vgg = base
        if use_final_conv == False:
            self.vgg.finalconv = None
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

    def forward(self, x):
        sources = list()
        x = self.quant(x)
        for k in range(7):
            x = self.vgg.features[k](x)
        s = x
        sources.append(s)
        for k in range(7, len(self.vgg.features)):
            x = self.vgg.features[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)
        output = []
        for idx, source in enumerate(sources):
            output.append(self.dequant_list[idx](source))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            None
        else:
            None

    def fuse_model(self):
        self.vgg.fuse_model()
        for v, extra in enumerate(self.extras):
            if v < len(self.extras) - 1:
                extra.fuse_model()


coco = {'num_classes': 201, 'lr_steps': (280000, 360000, 400000), 'max_iter': 400000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [21, 45, 99, 153, 207, 261], 'max_sizes': [45, 99, 153, 207, 261, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'COCO'}


voc = {'num_classes': 21, 'lr_steps': (80000, 100000, 120000), 'max_iter': 120000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 264, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'VOC'}


class SSD_MobileNetV2_HEAD(nn.Module):

    def __init__(self, base, extra_layers, cfg, extras_head_pos, num_classes, phase='train'):
        super(SSD_MobileNetV2_HEAD, self).__init__()
        self.data_cfg = (coco, voc)[num_classes == 21]
        self.phase = phase
        self.num_classes = num_classes
        self.loc_layers = []
        self.conf_layers = []
        self.priorbox = PriorBox(self.data_cfg)
        self.priors = self.priorbox.get_prior()
        self.priors.requires_grad = False
        self.loc_layers += [ConvBN(inp=base.features[6].conv[2].out_channels, oup=cfg[0] * 4, k_size=3, stride=1)]
        self.conf_layers += [ConvBN(inp=base.features[6].conv[2].out_channels, oup=cfg[0] * num_classes, k_size=3, stride=1)]
        self.loc_layers += [ConvBN(inp=base.features[-1][0].out_channels, oup=cfg[1] * 4, k_size=3, stride=1)]
        self.conf_layers += [ConvBN(inp=base.features[-1][0].out_channels, oup=cfg[1] * num_classes, k_size=3, stride=1)]
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        idx = 2
        for k, v in enumerate(extras_head_pos):
            try:
                self.loc_layers += [ConvBN(inp=extra_layers[v].out_channels, oup=cfg[idx] * 4, k_size=3, stride=1)]
                self.conf_layers += [ConvBN(inp=extra_layers[v].out_channels, oup=cfg[idx] * num_classes, k_size=3, stride=1)]
                idx += 1
            except:
                self.loc_layers += [ConvBN(inp=extra_layers[v - 1].out_channels, oup=cfg[idx] * 4, k_size=3, stride=1)]
                self.conf_layers += [ConvBN(inp=extra_layers[v - 1].out_channels, oup=cfg[idx] * num_classes, k_size=3, stride=1)]
                idx += 1
        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)

    def forward(self, sources):
        loc = []
        conf = []
        for x, l, c in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect.detect(loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes)), self.priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
        return output


class SegmentationLoss(nn.Module):

    def __init__(self, n_classes=21, loss_type='ce', device='cuda', ignore_idx=255, class_wts=None):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.device = device
        self.ignore_idx = ignore_idx
        self.smooth = 1e-06
        self.class_wts = class_wts
        if self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_wts)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=self.class_wts)

    def convert_to_one_hot(self, x):
        n, h, w = x.size()
        x[x == self.ignore_idx] = self.n_classes
        x = x.unsqueeze(1)
        x_one_hot = torch.zeros(n, self.n_classes + 1, h, w)
        x_one_hot = x_one_hot.scatter_(1, x, 1)
        return x_one_hot[:, :self.n_classes, :, :].contiguous()

    def forward(self, inputs, target):
        if isinstance(inputs, tuple):
            tuple_len = len(inputs)
            assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                if target.dim() == 3 and self.loss_type == 'bce':
                    target = self.convert_to_one_hot(target)
                loss_ = self.loss_fn(inputs[i], target)
                loss += loss_
        else:
            if target.dim() == 3 and self.loss_type == 'bce':
                target = self.convert_to_one_hot(target)
            return self.loss_fn(inputs, target)
        return loss


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        None

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and key == 'num_batches_tracked':
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                None
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        None
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    None
                None
        None

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, padding=0, groups=1, bias=False):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), groups=groups, bias=False)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, padding=0, groups=1, bias=False):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.cbr = nn.Sequential(nn.Conv2d(nIn, nOut, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), groups=groups, bias=False), nn.BatchNorm2d(nOut, eps=0.001), nn.ReLU(inplace=False))

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.cbr(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0', '1', '2'], inplace=True)


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kernel_size, stride=stride, padding=padding, bias=False, dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.cbr = CBR(nOut, nOut, 1, 1)
        self.quant_cat = nn.quantized.FloatFunctional()
        self.quant_add2 = nn.quantized.FloatFunctional()
        self.quant_add3 = nn.quantized.FloatFunctional()
        self.quant_add4 = nn.quantized.FloatFunctional()

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = self.quant_add2.add(add1, d4)
        add3 = self.quant_add3.add(add2, d8)
        add4 = self.quant_add4.add(add3, d16)
        combine = self.quant_cat.cat([d1, add1, add2, add3, add4], 1)
        output = combine
        output = self.cbr(combine)
        return output

    def fuse_model(self):
        self.cbr.fuse_model()


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.add = add
        if self.add:
            self.skip_add = nn.quantized.FloatFunctional()
        self.cbr = CBR(nOut, nOut, 1, 1)
        self.quant_cat = nn.quantized.FloatFunctional()
        self.quant_add2 = nn.quantized.FloatFunctional()
        self.quant_add3 = nn.quantized.FloatFunctional()
        self.quant_add4 = nn.quantized.FloatFunctional()

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = self.quant_add2.add(add1, d4)
        add3 = self.quant_add3.add(add2, d8)
        add4 = self.quant_add4.add(add3, d16)
        combine = self.quant_cat.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = self.skip_add.add(input, combine)
        output = self.cbr(combine)
        return output

    def fuse_model(self):
        self.cbr.fuse_model()


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=20, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = CBR(16 + 3, 16 + 3, 1, 1)
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = CBR(128 + 3, 128 + 3, 1, 1)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = CBR(256, 256, 1, 1)
        self.classifier = C(256, classes, 1, 1)
        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional()

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(self.quant_cat1.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(self.quant_cat2.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.b3(self.quant_cat3.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        return classifier

    def fuse_model(self):
        self.level1.fuse_model()
        self.level2_0.fuse_model()
        for i, layer in enumerate(self.level2):
            layer.fuse_model()
        self.level3_0.fuse_model()
        for i, layer in enumerate(self.level3):
            layer.fuse_model()
        self.b1.fuse_model()
        self.b2.fuse_model()
        self.b3.fuse_model()


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, padding=0, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.cb = nn.Sequential(nn.Conv2d(nIn, nOut, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), groups=groups, bias=False), nn.BatchNorm2d(nOut, eps=0.001))

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.cb(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)


class EESP(nn.Module):
    """
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kernel_size=3, stride=stride, groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = CBR(nOut, nOut, 1, 1)
        self.module_act = nn.ReLU(nOut)
        self.act_out = nOut
        self.downAvg = True if down_method == 'avg' else False
        self.quant_cat = nn.quantized.FloatFunctional()
        self.skip_add = nn.quantized.FloatFunctional()
        for i in range(1, k):
            exec('self.quant_add' + str(i) + '=nn.quantized.FloatFunctional()')

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            exec('out_k=self.quant_add' + str(k) + '.add(out_k, output[k - 1])')
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(self.quant_cat.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == input.size():
            expanded = self.skip_add.add(expanded, input)
        return self.module_act(expanded)

    def fuse_model(self):
        self.proj_1x1.fuse_model()
        self.conv_1x1_exp.fuse_model()
        self.br_after_cat.fuse_model()


class DownSampler(nn.Module):
    """
    Down-sampling fucntion that has two parallel branches: (1) avg pooling
    and (2) EESP block with stride of 2. The output feature maps of these branches
    are then concatenated and thresholded using an activation function (PReLU in our
    case) to produce the final output.
    """

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        """
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.reinf = reinf
        if reinf:
            self.inp_reinf = nn.Sequential(CBR(config_inp_reinf, config_inp_reinf, 3, 1), CB(config_inp_reinf, nout, 1, 1))
        self.act = nn.ReLU(nout)
        self.act_out = nout
        self.quant_cat = nn.quantized.FloatFunctional()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, input, input2=None):
        """
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        """
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = self.quant_cat.cat([avg_out, eesp_out], 1)
        if input2 is not None:
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = self.skip_add.add(output, self.inp_reinf(input2))
        return self.act(output)

    def fuse_model(self):
        self.eesp.fuse_model()
        if self.reinf:
            self.inp_reinf[0].fuse_model()
            self.inp_reinf[1].fuse_model()


class EESPNet(nn.Module):
    """
    This class defines the ESPNetv2 architecture for the ImageNet classification
    """

    def __init__(self, args):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param s: factor that scales the number of output feature maps
        """
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        classes = args.num_classes
        s = args.s
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            ValueError('Configuration not supported')
        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'
        self.level1 = CBR(channels, config[0], 3, 2)
        self.level1_act_out = config[0]
        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=r_lim[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=r_lim[1], reinf=self.input_reinforcement)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2], r_lim=r_lim[2]))
        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=r_lim[2], reinf=self.input_reinforcement)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3], r_lim=r_lim[3]))
        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=r_lim[3])
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4], r_lim=r_lim[4]))
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, seg=True):
        """
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None
        out_l2 = self.level2_0(out_l1, input)
        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        if not seg:
            out_l5_0 = self.level5_0(out_l4)
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)
            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=p, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)
            return self.classifier(output_1x1)
        return out_l1, out_l2, out_l3, out_l4

    def fuse_model(self, seg=True):
        self.level1.fuse_model()
        self.level2_0.fuse_model()
        self.level3_0.fuse_model()
        for i, layer in enumerate(self.level3):
            layer.fuse_model()
        self.level4_0.fuse_model()
        for i, layer in enumerate(self.level4):
            layer.fuse_model()
        if not seg:
            self.level5_0.fuse_model()
            for i, layer in enumerate(self.level5):
                layer.fuse_model()


class ESPNet(nn.Module):

    def __init__(self, classes=20, p=2, q=3):
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.b = CB(classes, classes, 1, 1)
        self.conv = CBR(19 + classes, classes, 3, 1)
        self.up_l3 = CBR(classes, classes, 1, stride=1, padding=0, bias=False)
        self.combine_l2_l3 = DilatedParllelResidualBlockB(2 * classes, classes, add=False)
        self.up_l2 = CBR(classes, classes, 1, stride=1, padding=0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional()
        self.quant_cat4 = nn.quantized.FloatFunctional()
        self.quant_cat5 = nn.quantized.FloatFunctional()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        input = self.quant(input)
        output0 = self.encoder.level1(input)
        inp1 = self.encoder.sample1(input)
        inp2 = self.encoder.sample2(input)
        output0_cat = self.encoder.b1(self.quant_cat1.cat([output0, inp1], 1))
        output1_0 = self.encoder.level2_0(output0_cat)
        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.encoder.b2(self.quant_cat2.cat([output1, output1_0, inp2], 1))
        output2_0 = self.encoder.level3_0(output1_cat)
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.encoder.b3(self.quant_cat3.cat([output2_0, output2], 1))
        l3 = self.upsample(self.b(self.encoder.classifier(output2_cat)))
        output2_c = self.up_l3(l3)
        output1_C = self.level3_C(output1_cat)
        l2 = self.upsample(self.combine_l2_l3(self.quant_cat4.cat([output1_C, output2_c], 1)))
        comb_l2_l3 = self.up_l2(l2)
        concat_features = self.conv(self.quant_cat5.cat([comb_l2_l3, output0_cat], 1))
        concat_features = self.upsample(concat_features)
        classifier = concat_features
        classifier = self.dequant(classifier)
        return classifier

    def fuse_model(self):
        self.encoder.level1.fuse_model()
        self.encoder.level2_0.fuse_model()
        for i, layer in enumerate(self.encoder.level2):
            layer.fuse_model()
        self.encoder.level3_0.fuse_model()
        for i, layer in enumerate(self.encoder.level3):
            layer.fuse_model()
        self.up_l3.fuse_model()
        self.up_l2.fuse_model()
        self.conv.fuse_model()
        self.combine_l2_l3.fuse_model()
        self.encoder.b1.fuse_model()
        self.encoder.b2.fuse_model()
        self.encoder.b3.fuse_model()
        self.b.fuse_model()


class ESPNetSeg(nn.Module):

    def __init__(self, classes=20, p=2, q=3):
        super().__init__()
        self.quantized = ESPNet(classes, p, q)
        self.classifier = C(classes, classes, 1, stride=1, padding=0)

    def forward(self, input):
        x = self.quantized(input)
        x = self.classifier(x)
        return x


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([C(features, features, 3, 1, groups=features) for size in sizes])
        self.project = CBR(features * (len(sizes) + 1), out_features, 1, 1)
        self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(self.quant_cat.cat(out, dim=1))

    def fuse_model(self):
        self.project.fuse_model()


class ESPNetv2Segmentation(nn.Module):
    """
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    """

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()
        classificationNet = EESPNet(args)
        self.net = classificationNet
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional()
        if args.s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        self.proj_L4_C = CBR(self.net.level4[-1].act_out, self.net.level3[-1].act_out, 1, 1)
        pspSize = 2 * self.net.level3[-1].act_out
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize // 2, stride=1, k=4, r_lim=7), PSPModule(pspSize // 2, pspSize // 2))
        self.project_l3 = CBR(pspSize // 2, classes, 1, 1)
        self.act_l3 = CBR(classes, classes, 1, 1)
        self.project_l2 = CBR(self.net.level2_0.act_out + classes, classes, 1, 1)
        self.init_params()

    def forward(self, x):
        """
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        x = self.quant(x)
        out_l1, out_l2, out_l3, out_l4 = self.net(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(self.quant_cat1.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(self.quant_cat2.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.dequant(self.quant_cat3.cat([out_l1, out_up_l2], 1))
        return output

    def fuse_model(self):
        self.net.fuse_model(seg=True)
        self.proj_L4_C.fuse_model()
        self.pspMod[0].fuse_model()
        self.pspMod[1].fuse_model()
        self.act_l3.fuse_model()
        self.project_l3.fuse_model()
        self.project_l2.fuse_model()

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ESPNetv2Seg(nn.Module):

    def __init__(self, args, classes=20, dataset='pascal'):
        super().__init__()
        self.quantized = ESPNetv2Segmentation(args, classes=classes, dataset=dataset)
        self.classifier = C(self.quantized.net.level1_act_out + classes, classes, 1, 1)

    def forward(self, input):
        x = self.quantized(input)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class _Head(nn.Module):

    def __init__(self, nclass, in_channels, inter_channels, dataset='city', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        atrous_rates = 6, 12, 18
        self.aspp = ASPP(in_channels, atrous_rates, norm_layer, **kwargs)
        self.auxlayer = _ConvBNReLU(inter_channels, 48, 1, 1)
        self.project = _ConvBNReLU(304, 256, 3, 3)
        self.reduce_conv = nn.Conv2d(256, nclass, 1, 1)
        self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, c1, c4):
        c4 = self.aspp(c4)
        c4 = F.interpolate(c4, c1.size()[2:], mode='bilinear', align_corners=True)
        c1 = self.auxlayer(c1)
        out = self.quant_cat.cat([c1, c4], dim=1)
        out = self.project(out)
        out = self.reduce_conv(out)
        return out

    def fuse_model(self):
        self.aspp.fuse_model()
        self.project.fuse_model()
        self.auxlayer.fuse_model()


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, dataset, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 256 // 2
        self.b0 = _ConvBNReLU(in_channels, out_channels, 1, 1)
        if dataset == 'city':
            self.b1 = nn.Sequential(nn.AvgPool2d((37, 37), (12, 12)), _ConvBN(in_channels, out_channels, 1, 1, bias=False), _Hsigmoid())
        else:
            self.b1 = nn.Sequential(nn.AvgPool2d((25, 25), (8, 8)), _ConvBN(in_channels, out_channels, 1, 1, bias=False), _Hsigmoid())
        self.quant_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = self.quant_mul.mul(feat1, feat2)
        return x

    def fuse_model(self):
        self.b0.fuse_model()
        self.b1[1].fuse_model()


class _AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), _ConvBNReLU(in_channels, out_channels, 1))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

    def fuse_model(self):
        self.gap[1].fuse_model()


class _RASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_RASPP, self).__init__()
        out_channels = 256
        self.b0 = _ConvBNReLU(in_channels, out_channels, 1)
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate1, dilation=rate1, norm_layer=norm_layer)
        self.b2 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate2, dilation=rate2, norm_layer=norm_layer)
        self.b3 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate3, dilation=rate3, norm_layer=norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
        self.project = nn.Sequential(_ConvBNReLU(5 * out_channels, out_channels, 1), nn.Dropout2d(0.1))
        self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = self.quant_cat.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

    def fuse_model(self):
        self.b0.fuse_model()
        self.b1.fuse_model()
        self.b2.fuse_model()
        self.b3.fuse_model()
        self.b4.fuse_model()
        self.project[0].fuse_model()


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(_ConvBNReLU(in_channels, inter_channels, 3, padding=1), nn.Dropout(0.1), nn.Conv2d(inter_channels, channels, 1))

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        self.block[0].fuse_model()


class _DWConvBNReLU(nn.Module):
    """Depthwise Separable Convolution in MobileNet.
    depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(_ConvBNReLU(in_channels, dw_channels, 3, stride, dilation, dilation, in_channels, norm_layer=norm_layer), _ConvBNReLU(dw_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()


class hard_sigmoid(nn.Module):
    """
    This class defines the ReLU6 activation to replace sigmoid
    """

    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=False)
        self.quant_mul = nn.quantized.FloatFunctional()
        self.quant_add = nn.quantized.FloatFunctional()

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = input
        output = self.quant_add.add_scalar(output, 3)
        output = self.relu6(output)
        output = self.quant_mul.mul_scalar(output, 1 / 6)
        return output


class CDilatedB(nn.Module):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.cb = nn.Sequential(nn.Conv2d(nIn, nOut, kernel_size, stride=stride, padding=padding, bias=False, dilation=d, groups=groups), nn.BatchNorm2d(nOut))

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.cb(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)


class CDilatedBR(nn.Module):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.cbr = nn.Sequential(nn.Conv2d(nIn, nOut, kernel_size, stride=stride, padding=padding, bias=False, dilation=d, groups=groups), nn.BatchNorm2d(nOut), nn.ReLU(inplace=False))

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.cbr(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0', '1', '2'], inplace=True)


class Shuffle(nn.Module):
    """
    This class implements Channel Shuffling
    """

    def __init__(self, groups):
        """
        :param groups: # of groups for shuffling
        """
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class DWSepConv(nn.Module):

    def __init__(self, nin, nout):
        super(DWSepConv, self).__init__()
        self.dwc = CBR(nin, nin, kernel_size=3, stride=1, groups=nin)
        self.pwc = CBR(nin, nout, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        x = self.dwc(x)
        x = self.pwc(x)
        return x

    def fuse_model(self):
        self.dwc.fuse_model()
        self.pwc.fuse_model()


class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)
        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(C(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))
        self.merge_layer = nn.Sequential(CBR(proj_planes * len(scales), proj_planes * len(scales), 1, 1), Shuffle(groups=len(scales)), CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes))
        if last_layer_br:
            self.br = CBR(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br)
        else:
            self.br = nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br)
        self.last_layer_br = last_layer_br
        self.scales = scales
        self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)
        out = self.quant_cat.cat(hs, dim=1)
        out = self.merge_layer(out)
        return self.br(out)

    def fuse_model(self):
        self.projection_layer.fuse_model()
        self.merge_layer[0].fuse_model()
        self.merge_layer[2].fuse_model()
        if self.last_layer_br:
            self.br.fuse_model()


class _MobileNetV2Seg(BaseModel):

    def __init__(self, nclass, backbone, pretrained_base=False, dataset='city', width_multi=1.0, **kwargs):
        super(_MobileNetV2Seg, self).__init__(nclass, backbone, pretrained_base, **kwargs)
        self.width_multi = width_multi
        in_channels = int(320 // 2 * width_multi)
        inter_channels = int(24 * width_multi)
        self.head = _Head(nclass, in_channels, inter_channels, dataset=dataset, width_multi=width_multi, **kwargs)
        self.quant = QuantStub()
        self.dequant1 = DeQuantStub()
        self.dequant2 = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        c1, c2, c3, c4 = self.base_forward(x)
        c1, c4 = self.head(c1, c4)
        c1 = self.dequant1(c1)
        c4 = self.dequant2(c4)
        return c1, c4

    def fuse_model(self):
        self.pretrained.fuse_model()
        self.head.fuse_model()


class MobileNetV2Seg(nn.Module):

    def __init__(self, nclass, backbone, pretrained_base=False, dataset='city', width_multi=1.0, **kwargs):
        super(MobileNetV2Seg, self).__init__()
        in_channels = int(320 // 2 * width_multi)
        inter_channels = int(24 * width_multi)
        self.quantized = _MobileNetV2Seg(nclass, backbone, pretrained_base, dataset, width_multi, **kwargs)
        self.project = nn.Conv2d(256 // 2, nclass, 1, 1)
        self.auxlayer = nn.Conv2d(inter_channels, nclass, 1, 1)

    def forward(self, x):
        size = x.size()[2:]
        c1, c4 = self.quantized(x)
        c4 = self.project(c4)
        c1 = self.auxlayer(c1)
        out = torch.add(c1, c4)
        x = F.interpolate(out, size, mode='bilinear', align_corners=True)
        return x


class _MobileNetV3Seg(BaseModel):

    def __init__(self, nclass, backbone, pretrained_base=False, dataset='city', crop_scale=1.0, **kwargs):
        super(_MobileNetV3Seg, self).__init__(nclass, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]
        in_channels = 960 // 2 if mode == 'large' else 576 // 2
        inter_channels = 40 if mode.startswith('large') else 24
        self.head = _Head(nclass, in_channels, inter_channels, mode=mode, dataset=dataset, **kwargs)
        self.quant = QuantStub()
        self.dequant1 = DeQuantStub()
        self.dequant2 = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        c1, _, _, c4 = self.base_forward(x)
        c1, c4 = self.head(c1, c4)
        c1 = self.dequant1(c1)
        c4 = self.dequant2(c4)
        return c1, c4

    def fuse_model(self):
        self.pretrained.fuse_model()
        self.head.fuse_model()


class MobileNetV3Seg(nn.Module):

    def __init__(self, nclass, backbone, pretrained_base=False, dataset='city', crop_scale=1.0, **kwargs):
        super(MobileNetV3Seg, self).__init__()
        mode = backbone.split('_')[-1]
        in_channels = 960 // 2 if mode == 'large' else 576 // 2
        inter_channels = 40 if mode.startswith('large') else 24
        self.quantized = _MobileNetV3Seg(nclass, backbone, pretrained_base, dataset, **kwargs)
        self.project = nn.Conv2d(256 // 2, nclass, 1, 1)
        self.auxlayer = nn.Conv2d(inter_channels, nclass, 1, 1)

    def forward(self, x):
        size = x.size()[2:]
        c1, c4 = self.quantized(x)
        c4 = self.project(c4)
        c1 = self.auxlayer(c1)
        out = torch.add(c1, c4)
        x = F.interpolate(out, size, mode='bilinear', align_corners=True)
        return x


class DataParallelModel(DataParallel):

    def forward(self, *inputs, **kwargs):
        """ The only difference between this and PyTorch's native implementation is that
        we do not need gather function because we will perform gathering inside Criterian
        wrapper."""
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        return self.parallel_apply(replicas, inputs, kwargs)


def parallel_apply_criteria(modules, inputs, targets, kwargs_tup=None, devices=None):
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
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.device(device):
                if isinstance(input, (list, tuple)):
                    input = input[0]
                if isinstance(target, (list, tuple)):
                    target = target[0]
                assert target.device == input.device
                if module.device != input.device:
                    module = module
                output = module(input, target, **kwargs)
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
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriteria(DataParallel):

    def forward(self, inputs, *targets, **kwargs):
        """
        Input is already sliced, so slice only target
        """
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = parallel_apply_criteria(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class QAdam(Optimizer):
    """Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, clip_by=0.001, toss_coin=True, noise_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, clip_by=clip_by, toss_coin=toss_coin, noise_decay=noise_decay)
        self.is_warmup = True
        super(QAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                toss_coin = group['toss_coin']
                noise_decay = group['noise_decay']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['restart_step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_min'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_max'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if toss_coin:
                        state['coin_toss'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_min, exp_max = state['exp_min'], state['exp_max']
                if toss_coin:
                    coin_toss = state['coin_toss']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                clip_by = group['clip_by']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if self.is_warmup:
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1, new_min).div_(bias_correction1)
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1, new_max).div_(bias_correction1)
                else:
                    state['restart_step'] += 1
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1, new_min).div_(bias_correction1)
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1, new_max).div_(bias_correction1)
                    noise_scale = (1 - noise_decay) ** state['restart_step']
                    grad_sensitivity = (exp_max - exp_min) * noise_scale
                    noise = np.random.laplace(0.0, 1.0, p.data.size())
                    noise = np.abs(noise)
                    noise = torch.from_numpy(noise).float()
                    sign = grad.sign()
                    noise.mul_(grad_sensitivity)
                    if toss_coin:
                        coin_toss.random_(2)
                        noise.mul_(coin_toss)
                    noise.mul_(sign)
                    if clip_by > 0.0:
                        noise.clamp_(-clip_by, clip_by)
                    grad.add_(noise)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.G_networks = [self.netG_A, self.netG_B]
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            if opt.q_optim:
                self.optimizer_G = QAdam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), clip_by=opt.clip_by)
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B']
        self.real_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ConvBNReLU(dim, dim, norm_layer=norm_layer, kernel_size=3, padding=p)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ConvBN(dim, dim, norm_layer=norm_layer, kernel_size=3, padding=p)]
        self.conv_block = nn.Sequential(*conv_block)
        self.skip_add = torch.nn.quantized.FloatFunctional()
        return self.conv_block

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.skip_add.add(x, self.conv_block(x))
        return out


class _ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(_ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        model = [ConvBNReLU(input_nc, ngf, norm_layer=norm_layer, kernel_size=7, padding=0)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ConvBNReLU(ngf * mult, ngf * mult * 2, norm_layer=norm_layer, kernel_size=3, stride=2, padding=1)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), ConvBNReLU(ngf * mult, int(ngf * mult / 2), norm_layer=norm_layer, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        input = self.quant(input)
        output = self.model(input)
        output = self.dequant(output)
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) in [ConvBNReLU, ConvBN]:
                m.fuse_model()


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        super(ResnetGenerator, self).__init__()
        self.padding = nn.ReflectionPad2d(3)
        self.quantized = _ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, padding_type=padding_type)
        model_fp = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model_fp += [nn.ReflectionPad2d(3)]
        model_fp += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_fp += [nn.Tanh()]
        self.model_fp = nn.Sequential(*model_fp)

    def forward(self, input):
        """Standard forward"""
        input = self.padding(input)
        output = self.quantized(input)
        output = self.model_fp(output)
        return output


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.G_networks = [self.netG]
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionL1 = torch.nn.L1Loss()
            if opt.q_optim:
                self.optimizer_G = QAdam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), clip_by=opt.clip_by)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B']
        self.real_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CascadePreExBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, quantized=False, kernel_size=3, stride=1, dilation=1, expand_ratio=6, reduce_factor=4, block_type='CAS'):
        super(CascadePreExBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.quantized = quantized
        if in_channels // reduce_factor < 8:
            block_type = 'MB'
        self.block_type = block_type
        r_channels = _make_divisible(in_channels // reduce_factor)
        if stride == 1 and in_channels == out_channels:
            self.reduction = False
        else:
            self.reduction = True
        if self.expand_ratio == 1:
            self.squeeze_conv = None
            self.conv1 = None
            n_channels = in_channels
        else:
            if block_type == 'CAS':
                self.squeeze_conv = ConvBNReLU(in_channels, r_channels, 1)
                n_channels = r_channels + in_channels
            else:
                n_channels = in_channels
            self.conv1 = ConvBNReLU(n_channels, n_channels * expand_ratio, 1)
        self.conv2 = ConvBNReLU(n_channels * expand_ratio, n_channels * expand_ratio, kernel_size, stride, (kernel_size - 1) // 2, 1, groups=n_channels * expand_ratio)
        self.reduce_conv = ConvBN(n_channels * expand_ratio, out_channels, 1)
        if self.quantized:
            self.skip_add = nn.quantized.FloatFunctional()
            self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        if not self.expand_ratio == 1:
            if self.block_type == 'CAS':
                squeezed = self.squeeze_conv(x)
                if self.quantized:
                    out = self.quant_cat.cat([squeezed, x], 1)
                else:
                    out = torch.cat([squeezed, x], 1)
            else:
                out = x
            out = self.conv1(out)
        else:
            out = x
        out = self.conv2(out)
        out = self.reduce_conv(out)
        if not self.reduction:
            if self.quantized:
                out = self.skip_add.add(x, out)
            else:
                out = torch.add(x, out)
        return out


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        None
        return state_dict
    else:
        None
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


class FrostNet(nn.Module):

    def __init__(self, mode='large', width_mult=1.0, bottleneck=CascadePreExBottleneck, quantized=False, pretrained='', **kwargs):
        super(FrostNet, self).__init__()
        self.quantized = quantized
        if mode == 'large':
            layer1_setting = [[3, 16, 1, 1, 1], [3, 24, 6, 4, 2], [3, 24, 3, 4, 1]]
            layer2_setting = [[5, 40, 6, 4, 2], [3, 40, 3, 4, 1]]
            layer3_setting = [[5, 80, 6, 4, 2], [5, 80, 3, 4, 1], [5, 80, 3, 4, 1], [5, 96, 6, 4, 1], [5, 96, 3, 4, 1], [3, 96, 3, 4, 1], [3, 96, 3, 4, 1]]
            layer4_setting = [[5, 192, 6, 2, 2], [5, 192, 6, 4, 1], [5, 192, 6, 4, 1], [5, 192, 3, 4, 1], [5, 192, 3, 4, 1]]
            layer5_setting = [[5, 320, 6, 2, 1]]
        elif mode == 'base':
            layer1_setting = [[3, 16, 1, 1, 1], [5, 24, 6, 4, 2], [3, 24, 3, 4, 1]]
            layer2_setting = [[5, 40, 3, 4, 2], [5, 40, 3, 4, 1]]
            layer3_setting = [[5, 80, 3, 4, 2], [3, 80, 3, 4, 1], [5, 96, 3, 2, 1], [3, 96, 3, 4, 1], [5, 96, 3, 4, 1], [5, 96, 3, 4, 1]]
            layer4_setting = [[5, 192, 6, 2, 2], [5, 192, 3, 2, 1], [5, 192, 3, 2, 1], [5, 192, 3, 2, 1]]
            layer5_setting = [[5, 320, 6, 2, 1]]
        elif mode == 'small':
            layer1_setting = [[3, 16, 1, 1, 1], [5, 24, 3, 4, 2], [3, 24, 3, 4, 1]]
            layer2_setting = [[5, 40, 3, 4, 2]]
            layer3_setting = [[5, 80, 3, 4, 2], [5, 80, 3, 4, 1], [3, 80, 3, 4, 1], [5, 96, 3, 2, 1], [5, 96, 3, 4, 1], [5, 96, 3, 4, 1]]
            layer4_setting = [[5, 192, 6, 4, 2], [5, 192, 6, 4, 1], [5, 192, 6, 4, 1]]
            layer5_setting = [[5, 320, 6, 2, 1]]
        else:
            raise ValueError('Unknown mode.')
        self.in_channels = _make_divisible(int(32 * min(1.0, width_mult)))
        self.conv1 = ConvBNReLU(3, self.in_channels, 3, 2, 1)
        self.layer1 = self._make_layer(bottleneck, layer1_setting, width_mult, 1)
        self.layer2 = self._make_layer(bottleneck, layer2_setting, width_mult, 1)
        self.layer3 = self._make_layer(bottleneck, layer3_setting, width_mult, 1)
        self.layer4 = self._make_layer(bottleneck, layer4_setting, width_mult, 1)
        self.layer5 = self._make_layer(bottleneck, layer5_setting, width_mult, 1)
        last_in_channels = self.in_channels
        self.mode = mode

    def init_weights(self, pretrained):
        if pretrained != '':
            load_checkpoint(self, pretrained, use_ema=True, strict=False)
        else:
            None
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _make_layer(self, block, block_setting, width_mult, dilation=1):
        layers = list()
        for k, c, e, r, s in block_setting:
            out_channels = _make_divisible(int(c * width_mult))
            stride = s if dilation == 1 else 1
            layers.append(block(self.in_channels, out_channels, quantized=self.quantized, kernel_size=k, stride=s, dilation=dilation, expand_ratio=e, reduce_factor=r))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        features = [x1, x2, x3, x5]
        return features

    def _freeze_stages(self):
        """Freeze BatchNorm layers."""
        None
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (C,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CB,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBR,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CDilated,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CDilatedB,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CDilatedBR,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CascadePreExBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWSepConv,
     lambda: ([], {'nin': 4, 'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ESPNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ESPNetSeg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ESPNet_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EfficientPyrPool,
     lambda: ([], {'in_planes': 4, 'proj_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FrostNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionAux,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputProjectionA,
     lambda: ([], {'samplingTimes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MNASNet,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (PSPModule,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QSSD_TDSOD_Feat,
     lambda: ([], {'size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (QuantizableBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Shuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_AsppPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ConvBNHswish,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ConvBNReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DWConvBNReLU,
     lambda: ([], {'in_channels': 4, 'dw_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_FCNHead,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_InvertedResidual,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1, 'expansion_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_LRASPP,
     lambda: ([], {'in_channels': 4, 'norm_layer': 1, 'dataset': torch.rand([4, 4, 4, 4])}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (_RASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (baseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (conv_bn,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_bn_no_relu,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (downsample_0,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (downsample_1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (dwd_block,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hard_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (trans_block,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (upsample,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_clovaai_frostnet(_paritybench_base):
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

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

