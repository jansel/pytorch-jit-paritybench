import sys
_module = sys.modules[__name__]
del sys
build = _module
quantization_aware_training = _module
datasets = _module
cifar10 = _module
infer = _module
models = _module
models = _module
resnet = _module
parser = _module
setup = _module
train = _module
utils = _module
utilities = _module
generate_data = _module
optimize_detector = _module
optimize_recognizer = _module
run_end2end = _module
dump_converters = _module
profile_timm = _module
setup = _module
torch2trt = _module
contrib = _module
qat = _module
QuantConv = _module
QuantConvBN = _module
QuantRelu = _module
converters = _module
layers = _module
_utils = _module
quant_activation = _module
quant_conv = _module
AdaptiveAvgPool2d = _module
AdaptiveAvgPool3d = _module
BatchNorm1d = _module
BatchNorm2d = _module
BatchNorm3d = _module
Conv = _module
Conv1d = _module
Conv2d = _module
ConvTranspose = _module
ConvTranspose2d = _module
Linear = _module
LogSoftmax = _module
activation = _module
adaptive_avg_pool2d = _module
adaptive_avg_pool3d = _module
adaptive_max_pool2d = _module
adaptive_max_pool3d = _module
add = _module
avg_pool = _module
batch_norm = _module
cat = _module
chunk = _module
clamp = _module
clone = _module
compare = _module
conv_functional = _module
div = _module
dummy_converters = _module
einsum = _module
example_plugin = _module
expand = _module
flatten = _module
floordiv = _module
gelu = _module
getitem = _module
getitem_test = _module
group_norm = _module
identity = _module
instance_norm = _module
interpolate = _module
layer_norm = _module
matmul = _module
max = _module
max_pool1d = _module
max_pool2d = _module
max_pool3d = _module
mean = _module
min = _module
mod = _module
mul = _module
narrow = _module
ne = _module
normalize = _module
pad = _module
permute = _module
pow = _module
prelu = _module
prod = _module
reflection_pad_2d = _module
relu = _module
relu6 = _module
roll = _module
sigmoid = _module
silu = _module
softmax = _module
split = _module
squeeze = _module
stack = _module
sub = _module
sum = _module
tanh = _module
tensor = _module
transpose = _module
unary = _module
unsqueeze = _module
view = _module
dataset = _module
dataset_calibrator = _module
dataset_calibrator_test = _module
dataset_test = _module
dynamic_shape_test = _module
flatten_module = _module
flatten_module_test = _module
flattener = _module
flattener_test = _module
module_test = _module
test = _module
tests = _module
test_contiguous = _module
test_flatten_dynamic = _module
test_interpolate_dynamic = _module
test_legacy_max_batch_size = _module
test_tensor_ne = _module
test_tensor_shape = _module
test_tensor_shape_div_batch = _module
timm = _module
test_maxvit = _module
torchvision = _module
classification = _module
save_load = _module
segmentation = _module
torch2trt = _module

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


from string import Template


import torch


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


import numpy as np


import torch.optim as optim


import time


import collections


import torchvision.models as models


import re


import math


from typing import Literal


from enum import Enum


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


import copy


import inspect


from torch import nn


import torch.nn.functional as F


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.modules.conv import _ConvTransposeNd


from uuid import uuid1


from collections import defaultdict


class qconv2d(torch.nn.Module):
    """
    common layer for qat and non qat mode
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=0, groups: int=1, dilation: int=1, bias=None, padding_mode: str='zeros', eps: float=1e-05, momentum: float=0.1, freeze_bn=False, act: bool=True, norm: bool=True, qat: bool=False, infer: bool=False):
        super().__init__()
        if qat:
            if infer:
                if norm:
                    layer_list = [IQuantConvBN2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode)]
                else:
                    layer_list = [IQuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode)]
            elif norm:
                layer_list = [QuantConvBN2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode, quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
            else:
                layer_list = [QuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias, padding_mode=padding_mode, quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)]
            if act:
                if infer:
                    layer_list.append(IQuantReLU())
                else:
                    layer_list.append(QuantReLU())
            self.qconv = nn.Sequential(*layer_list)
        else:
            layer_list = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias, groups=groups)]
            if norm:
                layer_list.append(nn.BatchNorm2d(out_channels))
            if act:
                layer_list.append(nn.ReLU())
            self.qconv = nn.Sequential(*layer_list)

    def forward(self, inputs):
        return self.qconv(inputs)


class vanilla_cnn(nn.Module):

    def __init__(self, qat_mode=False, infer=False):
        super().__init__()
        self.qat = qat_mode
        self.layer1 = qconv2d(3, 32, padding=1, qat=qat_mode, infer=infer)
        self.layer2 = qconv2d(32, 64, padding=1, qat=qat_mode, infer=infer)
        self.layer3 = qconv2d(64, 128, padding=1, qat=qat_mode, infer=infer)
        self.layer4 = qconv2d(128, 256, padding=1, qat=qat_mode, infer=infer)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=8)
        self.fcs = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, norm=False, act=False, qat_mode=False, infer=False):
    """3x3 convolution with padding"""
    return qconv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, groups=groups, padding=dilation, dilation=dilation, bias=False, act=act, norm=norm, qat=qat_mode, infer=infer)


class qrelu(torch.nn.Module):

    def __init__(self, inplace=False, qat=False, infer=False):
        super().__init__()
        if qat:
            if infer:
                self.relu = IQuantReLU(inplace)
            else:
                self.relu = QuantReLU(inplace)
        else:
            self.relu = nn.ReLU(inplace)

    def forward(self, input):
        return self.relu(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, qat_mode=False, infer=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, norm=True, qat_mode=qat_mode, infer=infer)
        self.relu1 = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.conv2 = conv3x3(planes, planes, norm=True, qat_mode=qat_mode, infer=infer)
        self.relu2 = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


def conv1x1(in_planes, out_planes, stride=1, norm=False, act=False, qat_mode=False, infer=False):
    """1x1 convolution"""
    return qconv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False, act=act, norm=norm, qat=qat_mode, infer=infer)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, qat_mode=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, qat_mode=qat_mode)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, qat_mode=qat_mode)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, qat_mode=qat_mode)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, qat_mode=False, infer=False):
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
        self.conv1 = qconv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, norm=True, act=False, qat=qat_mode, infer=infer)
        self.relu = qrelu(inplace=True, qat=qat_mode, infer=infer)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], qat_mode=qat_mode, infer=infer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], qat_mode=qat_mode, infer=infer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], qat_mode=qat_mode, infer=infer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], qat_mode=qat_mode, infer=infer)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, qat_mode=False, infer=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride, norm=True, qat_mode=qat_mode, infer=infer))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, qat_mode, infer=infer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, qat_mode=qat_mode, infer=infer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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


class PoolFix(torch.nn.Module):

    def forward(self, x):
        return torch.mean(x, dim=-1, keepdim=True)


class TensorQuantizer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('learned_amax', torch.tensor(1.0))


class Add(torch.nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


class IAdd(torch.nn.Module):

    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


class TorchAdd(torch.nn.Module):

    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


class RAddInt(torch.nn.Module):

    def __init__(self):
        super(RAddInt, self).__init__()

    def forward(self, x):
        return 1 + x


class RAddFloat(torch.nn.Module):

    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        return 1.0 + x


class AddConstantNoBatch(torch.nn.Module):

    def __init__(self):
        super(AddConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x + self.y


class AddConstantBatch(torch.nn.Module):

    def __init__(self):
        super(AddConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x + self.y


class Cat(torch.nn.Module):

    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)


class TorchChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)


class TensorChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)


class TorchClampMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp_min(x, -0.1)


class TensorClampMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp_min(-0.1)


class TorchClampMax(torch.nn.Module):

    def forward(self, x):
        return torch.clamp_max(x, 0.1)


class TensorClampMax(torch.nn.Module):

    def forward(self, x):
        return x.clamp_max(0.1)


class TorchClamp(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, -0.1, 0.1)


class TensorClamp(torch.nn.Module):

    def forward(self, x):
        return x.clamp(-0.1, 0.1)


class TorchClampOptionMax(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, max=0.1)


class TorchClampOptionMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.1)


class TorchClampOptionMaxMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.1, max=0.1)


class TensorClampOptionMax(torch.nn.Module):

    def forward(self, x):
        return x.clamp(max=0.1)


class TensorClampOptionMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp(min=-0.1)


class TensorClampOptionMaxMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp(min=-0.1, max=0.1)


class Clone(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


class TorchClone(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clone(x)


class GT(torch.nn.Module):

    def __init__(self):
        super(GT, self).__init__()

    def forward(self, x, y):
        return x > y


class LT(torch.nn.Module):

    def __init__(self):
        super(LT, self).__init__()

    def forward(self, x, y):
        return x < y


class EQ(torch.nn.Module):

    def __init__(self):
        super(EQ, self).__init__()

    def forward(self, x, y):
        return x == y


class FunctionalConv2d(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return x


class FunctionalConv3d(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv3d(*args, **kwargs)

    def forward(self, x):
        x = torch.nn.functional.conv3d(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return x


class Div(torch.nn.Module):

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x, y):
        return x / y


class IDiv(torch.nn.Module):

    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x


class TorchDiv(torch.nn.Module):

    def __init__(self):
        super(TorchDiv, self).__init__()

    def forward(self, x, y):
        return torch.div(x, y)


class RDivInt(torch.nn.Module):

    def __init__(self):
        super(RDivInt, self).__init__()

    def forward(self, x):
        return 100 / x


class RDivFloat(torch.nn.Module):

    def __init__(self):
        super(RDivFloat, self).__init__()

    def forward(self, x):
        return 100.0 / x


class DivConstantNoBatch(torch.nn.Module):

    def __init__(self):
        super(DivConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x / self.y


class DivConstantBatch(torch.nn.Module):

    def __init__(self):
        super(DivConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x / self.y


class Einsum(nn.Module):

    def __init__(self, einsum_eq):
        super().__init__()
        self.einsum_eq = einsum_eq

    def forward(self, *args):
        return torch.einsum(self.einsum_eq, *args)


class ExpandModule(torch.nn.Module):

    def __init__(self, *sizes):
        super(ExpandModule, self).__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.expand(*self.sizes)


class FloorDiv(torch.nn.Module):

    def __init__(self):
        super(FloorDiv, self).__init__()

    def forward(self, x, y):
        return x // y


class FloorDivAssign(torch.nn.Module):

    def __init__(self):
        super(FloorDivAssign, self).__init__()

    def forward(self, x, y):
        x //= y
        return x


class FloorDivConst(torch.nn.Module):

    def __init__(self):
        super(FloorDivConst, self).__init__()

    def forward(self, x):
        return x // 2.0


class TorchFloorDiv(torch.nn.Module):

    def __init__(self):
        super(TorchFloorDiv, self).__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


class LambdaModule(torch.nn.Module):

    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class YOLOXFocusTestModule(nn.Module):

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return x


class Interpolate(torch.nn.Module):

    def __init__(self, size=None, scale_factor=None, mode=None, align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Matmul(torch.nn.Module):

    def forward(self, x, y):
        return x @ y


class MaxElementwise(torch.nn.Module):

    def forward(self, x, y):
        return torch.max(x, y)


class MaxPool1D(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return torch.nn.functional.max_pool1d(x, self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)


class Mean(torch.nn.Module):

    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(self.dim, self.keepdim)


class MinElementwise(torch.nn.Module):

    def forward(self, x, y):
        return torch.min(x, y)


class Mod(torch.nn.Module):

    def __init__(self):
        super(Mod, self).__init__()

    def forward(self, x, y):
        return x % y


class ModAssign(torch.nn.Module):

    def __init__(self):
        super(ModAssign, self).__init__()

    def forward(self, x, y):
        x %= y
        return x


class ModConst(torch.nn.Module):

    def __init__(self):
        super(ModConst, self).__init__()

    def forward(self, x):
        return x % 2.0


class TorchMod(torch.nn.Module):

    def __init__(self):
        super(TorchMod, self).__init__()

    def forward(self, x, y):
        return torch.fmod(x, y)


class Mul(torch.nn.Module):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y


class IMul(torch.nn.Module):

    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


class TorchMul(torch.nn.Module):

    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


class RMulInt(torch.nn.Module):

    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


class RMulFloat(torch.nn.Module):

    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


class MulConstantNoBatch(torch.nn.Module):

    def __init__(self):
        super(MulConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x * self.y


class MulConstantBatch(torch.nn.Module):

    def __init__(self):
        super(MulConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x * self.y


class Narrow(torch.nn.Module):

    def __init__(self, dim, start, length):
        super(Narrow, self).__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x):
        return torch.narrow(x, self.dim, self.start, self.length)


class NotEqual(torch.nn.Module):

    def __init__(self):
        super(NotEqual, self).__init__()

    def forward(self, x, y):
        return x != y


class NotEqualConst(torch.nn.Module):

    def __init__(self):
        super(NotEqualConst, self).__init__()

    def forward(self, x):
        return x != 13.62


class TorchNotEqual(torch.nn.Module):

    def __init__(self):
        super(TorchNotEqual, self).__init__()

    def forward(self, x, y):
        return torch.ne(x, y)


class Normalize(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.nn.functional.normalize(x, *self.args, **self.kwargs)


class Pad(torch.nn.Module):

    def __init__(self, pad):
        super(Pad, self).__init__()
        self.pad = pad

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad)


class Permute(torch.nn.Module):

    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args).contiguous()


class Pow(torch.nn.Module):

    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x, y):
        return x ** y


class TorchPow(torch.nn.Module):

    def __init__(self):
        super(TorchPow, self).__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


class RpowInt(torch.nn.Module):

    def __init__(self):
        super(RpowInt, self).__init__()

    def forward(self, x):
        return 2 ** x


class RpowFloat(torch.nn.Module):

    def __init__(self):
        super(RpowFloat, self).__init__()

    def forward(self, x):
        return 2.0 ** x


class FunctionalRelu(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.relu(x)


class TensorRelu(torch.nn.Module):

    def __init__(self):
        super(TensorRelu, self).__init__()

    def forward(self, x):
        return x.relu()


class FunctionalRelu6(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.relu6(x)


class Roll(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.roll(x, *self.args, **self.kwargs)


class TensorSigmoid(torch.nn.Module):

    def __init__(self):
        super(TensorSigmoid, self).__init__()

    def forward(self, x):
        return x.sigmoid()


class TorchSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TorchSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.split(x, *self.args, **self.kwargs)


class TensorSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TensorSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.split(*self.args, **self.kwargs)


class Squeeze(torch.nn.Module):

    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class Stack(torch.nn.Module):

    def __init__(self, dim):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.stack(x, dim=self.dim)


class Sub(torch.nn.Module):

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y


class ISub(torch.nn.Module):

    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


class TorchSub(torch.nn.Module):

    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


class RSubInt(torch.nn.Module):

    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


class RSubFloat(torch.nn.Module):

    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


class SubConstantNoBatch(torch.nn.Module):

    def __init__(self):
        super(SubConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x - self.y


class SubConstantBatch(torch.nn.Module):

    def __init__(self):
        super(SubConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x - self.y


class DisparityRegression(nn.Module):

    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.register_buffer('disp', torch.arange(maxdisp, dtype=torch.float32).view(maxdisp, 1, 1))

    def forward(self, x):
        return x * self.disp


class TorchTensor(torch.nn.Module):

    def __init__(self):
        super(TorchTensor, self).__init__()

    def forward(self, x):
        return x + torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=torch.device('cuda'))


class Transpose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()


class TensorTranspose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super(TensorTranspose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class UnaryModule(torch.nn.Module):

    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class UnSqueeze(torch.nn.Module):

    def __init__(self, dim):
        super(UnSqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)


class View(torch.nn.Module):

    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


class Unflatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.flatten(args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.unflatten(output)
        return output


class Flatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.unflatten(*args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.flatten(output)
        return output


class FlattenModule(torch.nn.Module):

    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


class ModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


def _default_condition(x):
    return isinstance(x, torch.Tensor) and (x.dtype is torch.half or x.dtype is torch.float or x.dtype == torch.bool)


def _make_schema_from_value(value, condition=_default_condition, size=0):
    if condition(value):
        return size, size + 1
    elif isinstance(value, list) or isinstance(value, tuple):
        schema = []
        for child_value in value:
            child_schema, size = _make_schema_from_value(child_value, condition, size)
            schema.append(child_schema)
        if isinstance(value, tuple):
            schema = tuple(schema)
        return schema, size
    elif isinstance(value, dict):
        schema = {}
        for child_key in sorted(value.keys()):
            child_value = value[child_key]
            child_schema, size = _make_schema_from_value(child_value, condition, size)
            schema[child_key] = child_schema
        return schema, size
    else:
        return None, size


class Flattener(object):

    def __init__(self, schema, size):
        self._schema = schema
        self._size = size

    @staticmethod
    def from_value(value, condition=_default_condition):
        return Flattener(*_make_schema_from_value(value, condition))

    @staticmethod
    def from_dict(x):
        return Flattener(x['schema'], x['size'])

    def dict(self):
        return {'schema': self.schema, 'size': self.size}

    @property
    def schema(self):
        return self._schema

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size

    def _flatten(self, value, result):
        if isinstance(self._schema, int):
            result[self._schema] = value
        elif isinstance(self._schema, list) or isinstance(self._schema, tuple):
            for child_value, child_schema in zip(value, self._schema):
                Flattener(child_schema, self.size)._flatten(child_value, result)
        elif isinstance(self._schema, dict):
            for key in sorted(self._schema.keys()):
                child_value = value[key]
                child_schema = self._schema[key]
                Flattener(child_schema, self.size)._flatten(child_value, result)

    def flatten(self, value):
        result = [None for i in range(self.size)]
        self._flatten(value, result)
        return result

    def unflatten(self, flattened):
        if isinstance(self._schema, int):
            return flattened[self._schema]
        elif isinstance(self._schema, list) or isinstance(self._schema, tuple):
            result = []
            for child_schema in self._schema:
                result.append(Flattener(child_schema, self.size).unflatten(flattened))
            if isinstance(self._schema, tuple):
                result = tuple(result)
            return result
        elif isinstance(self._schema, dict):
            result = {}
            for child_key in sorted(self._schema.keys()):
                child_schema = self._schema[child_key]
                result[child_key] = Flattener(child_schema, self.size).unflatten(flattened)
            return result
        else:
            return None


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def trt_version():
    return trt.__version__


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names
        state_dict[prefix + 'input_flattener'] = self.input_flattener.dict()
        state_dict[prefix + 'output_flattener'] = self.output_flattener.dict()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()
        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']
        if 'input_flattener' in state_dict:
            self.input_flattener = Flattener.from_dict(state_dict['input_flattener'])
        else:
            self.input_flattener = None
        if 'output_flattener' in state_dict:
            self.output_flattener = Flattener.from_dict(state_dict['output_flattener'])
        else:
            self.output_flattener = None

    def forward(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Add,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AddConstantBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (AddConstantNoBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Clone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DisparityRegression,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Div,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DivConstantBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (DivConstantNoBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (EQ,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FlattenModule,
     lambda: ([], {'start_dim': 4, 'end_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (FloorDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FloorDivAssign,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FloorDivConst,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FunctionalConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FunctionalConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FunctionalRelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FunctionalRelu6,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ISub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaModule,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Matmul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxElementwise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPool1D,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MinElementwise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mod,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModAssign,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModConst,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MulConstantBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (MulConstantNoBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NotEqual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NotEqualConst,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoolFix,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RAddFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RAddInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDivFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDivInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMulFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMulInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RSubFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RSubInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RpowFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RpowInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Squeeze,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Sub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SubConstantBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (SubConstantNoBatch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 10, 10])], {}),
     True),
    (TensorClamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMaxMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorRelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorTranspose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (TorchAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMaxMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchFloorDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchMod,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchNotEqual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchPow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchSub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (UnSqueeze,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnaryModule,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (YOLOXFocusTestModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (qconv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (qrelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIA_AI_IOT_torch2trt(_paritybench_base):
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

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

    def test_077(self):
        self._check(*TESTCASES[77])

    def test_078(self):
        self._check(*TESTCASES[78])

    def test_079(self):
        self._check(*TESTCASES[79])

    def test_080(self):
        self._check(*TESTCASES[80])

    def test_081(self):
        self._check(*TESTCASES[81])

    def test_082(self):
        self._check(*TESTCASES[82])

    def test_083(self):
        self._check(*TESTCASES[83])

