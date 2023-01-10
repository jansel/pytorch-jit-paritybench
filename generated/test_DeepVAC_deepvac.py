import sys
_module = sys.modules[__name__]
del sys
deepvac = _module
aug = _module
base_aug = _module
composer = _module
emboss_helper = _module
face_aug = _module
factory = _module
haishoku_helper = _module
line_helper = _module
perspective_helper = _module
remaper_helper = _module
seg_audit = _module
seg_aug = _module
text_aug = _module
warp_mls_helper = _module
yolo_aug = _module
backbones = _module
activate_layer = _module
bottleneck_layer = _module
conv_layer = _module
fpn_layer = _module
mobilenet = _module
module_helper = _module
op_layer = _module
regnet = _module
repvgg = _module
res_layer = _module
resnet = _module
se_layer = _module
spp_layer = _module
ssh_layer = _module
weights_init = _module
cast = _module
base = _module
cast_helper = _module
coreml = _module
mnn = _module
ncnn = _module
onnx = _module
script = _module
tensorrt = _module
tnn = _module
trace = _module
core = _module
config = _module
deepvac = _module
qat = _module
report = _module
datasets = _module
base_dataset = _module
coco = _module
file_line = _module
os_walk = _module
experimental = _module
distill = _module
feature = _module
vector = _module
loss = _module
face_loss = _module
loss = _module
utils = _module
annotation = _module
crypto = _module
ddp = _module
face_align = _module
face_utils = _module
log = _module
meter = _module
pallete = _module
time = _module
user_config = _module
version = _module
setup = _module

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


import random


from scipy import ndimage


import torch


import torch.nn as nn


from collections import OrderedDict


import torch.nn.functional as F


from typing import List


import time


import torch.optim as optim


from torchvision import transforms as trans


from torchvision.models import resnet50


import copy


from torch.quantization import quantize_dynamic_jit


from torch.quantization import per_channel_dynamic_qconfig


from torch.quantization import get_default_qconfig


from torch.quantization import quantize_jit


from torch.quantization import default_dynamic_qconfig


from torch.quantization import float_qparams_weight_only_qconfig


from torch.quantization.quantize_fx import prepare_fx


from torch.quantization.quantize_fx import convert_fx


from torch.quantization.quantize_fx import prepare_qat_fx


from torch.quantization import QuantStub


from torch.quantization import DeQuantStub


import collections


import math


from typing import Callable


import torch.distributed as dist


from torch.utils.data import DataLoader


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.utils.data import Dataset


from itertools import product as product


from math import ceil


class hswish(nn.Module):

    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = x * self.relu(x + 3) / 6
        return out


class hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x + 3) / 6
        return out


class Conv2dBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes, momentum=0.1), nn.ReLU(inplace=True))


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, outplanes: int, stride: int=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2dBNReLU(in_planes=inplanes, out_planes=outplanes, kernel_size=1)
        self.conv2 = Conv2dBNReLU(in_planes=outplanes, out_planes=outplanes, kernel_size=3, stride=stride)
        outplanes_after_expansion = outplanes * self.expansion
        self.conv3 = nn.Conv2d(outplanes, outplanes_after_expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outplanes_after_expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != outplanes_after_expansion:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, outplanes_after_expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outplanes_after_expansion))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class BottleneckIR(nn.Module):

    def __init__(self, inplanes: int, outplanes: int, stride: int):
        super(BottleneckIR, self).__init__()
        self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride) if inplanes == outplanes else nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outplanes))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(inplanes), nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(outplanes), nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(outplanes))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Conv2dBN(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBN, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes))


class Conv2dBNWithName(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNWithName, self).__init__(OrderedDict([('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)), ('bn', nn.BatchNorm2d(out_planes))]))


class Conv2dBNPReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNPReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes, momentum=0.1), nn.PReLU(out_planes))


class Conv2dBnAct(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False, act=True):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBnAct, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes), nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity())


class Conv2dBNHardswish(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHardswish, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes, momentum=0.1), nn.Hardswish())


class Conv2dBNHswish(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHswish, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes, momentum=0.1), hswish(inplace=True))


class DepthWiseConv2d(nn.Module):

    def __init__(self, inplanes: int, outplanes: int, kernel_size: int, stride: int, padding: int, groups: int, residual: bool=False, bias: bool=False):
        super(DepthWiseConv2d, self).__init__()
        self.conv1 = Conv2dBNPReLU(inplanes, groups, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dBNPReLU(groups, groups, kernel_size, stride, padding, groups)
        self.conv3 = nn.Conv2d(groups, outplanes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.residual = residual

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        if self.residual:
            x = identity + x
        return x


class Conv2dBNLeakyReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False, leaky=0):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNLeakyReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_planes), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class BNPReLU(nn.Sequential):

    def __init__(self, out_planes):
        super(BNPReLU, self).__init__(nn.BatchNorm2d(out_planes), nn.PReLU(out_planes))


class Conv2dDilatedBN(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, padding=None, groups=1, bias=False):
        padding = (kernel_size - 1) // 2 * dilation
        super(Conv2dDilatedBN, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_planes))


class FPN(nn.Module):

    def __init__(self, in_planes: list, out_planes: int):
        super(FPN, self).__init__()
        leaky = 0.1 if out_planes <= 64 else 0
        self.conv1 = Conv2dBNLeakyReLU(in_planes[0], out_planes, kernel_size=1, padding=0, leaky=leaky)
        self.conv2 = Conv2dBNLeakyReLU(in_planes[1], out_planes, kernel_size=1, padding=0, leaky=leaky)
        self.conv3 = Conv2dBNLeakyReLU(in_planes[2], out_planes, kernel_size=1, padding=0, leaky=leaky)
        self.conv4 = Conv2dBNLeakyReLU(out_planes, out_planes, padding=1, leaky=leaky)
        self.conv5 = Conv2dBNLeakyReLU(out_planes, out_planes, padding=1, leaky=leaky)

    def forward(self, input):
        output1 = self.conv1(input[0])
        output2 = self.conv2(input[1])
        output3 = self.conv3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.conv5(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.conv4(output1)
        out = [output1, output2, output3]
        return out


def makeDivisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, makeDivisible(channel // reduction, 8)), nn.ReLU(inplace=True), nn.Linear(makeDivisible(channel // reduction, 8), channel), hsigmoid(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, use_se, use_hs, padding=None):
        super(InvertedResidual, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        hidden_dim = makeDivisible(inp * expand_ratio, 8)
        assert stride in [1, 2, (2, 1)]
        assert kernel_size in [3, 5]
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dBNHswish(inp, hidden_dim, kernel_size=1))
        layers.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), hswish() if use_hs else nn.ReLU(inplace=True), SELayer(hidden_dim) if use_se else nn.Identity(), nn.Conv2d(hidden_dim, oup, kernel_size=1), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def initWeightsKaiming(civilnet):
    for m in civilnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MobileNetV3(nn.Module):

    def __init__(self, class_num=1000, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.width_mult = width_mult
        self.class_num = class_num
        self.auditConfig()
        input_channel = makeDivisible(16 * self.width_mult, 8)
        layers = [Conv2dBNHswish(3, input_channel, stride=2)]
        for k, t, c, use_se, use_hs, s in self.cfgs:
            exp_size = makeDivisible(input_channel * t, 8)
            output_channel = makeDivisible(c * self.width_mult, 8)
            layers.append(InvertedResidual(input_channel, output_channel, k, s, t, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv = Conv2dBNHswish(input_channel, exp_size, kernel_size=1)
        self.fc_inp = exp_size
        self.initFc()
        initWeightsKaiming(self)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def initFc(self):
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = makeDivisible(self.last_output_channel * self.width_mult, 8) if self.width_mult > 1.0 else self.last_output_channel
        self.classifier = nn.Sequential(nn.Linear(self.fc_inp, output_channel), hswish(), nn.Dropout(0.2), nn.Linear(output_channel, self.class_num))

    def auditConfig(self):
        self.last_output_channel = 1024
        self.cfgs = [[3, 1, 16, True, False, 2], [3, 4.5, 24, False, False, 2], [3, 3.67, 24, False, False, 1], [5, 4, 40, True, True, 2], [5, 6, 40, True, True, 1], [5, 6, 40, True, True, 1], [5, 3, 48, True, True, 1], [5, 3, 48, True, True, 1], [5, 6, 96, True, True, 2], [5, 6, 96, True, True, 1], [5, 6, 96, True, True, 1]]


class MobileNetV3Large(MobileNetV3):

    def auditConfig(self):
        self.last_output_channel = 1280
        self.cfgs = [[3, 1, 16, False, False, 1], [3, 4, 24, False, False, 2], [3, 3, 24, False, False, 1], [5, 3, 40, True, False, 2], [5, 3, 40, True, False, 1], [5, 3, 40, True, False, 1], [3, 6, 80, False, True, 2], [3, 2.5, 80, False, True, 1], [3, 2.3, 80, False, True, 1], [3, 2.3, 80, False, True, 1], [3, 6, 112, True, True, 1], [3, 6, 112, True, True, 1], [5, 6, 160, True, True, 2], [5, 6, 160, True, True, 1], [5, 6, 160, True, True, 1]]


class MobileNetV3OCR(MobileNetV3):

    def auditConfig(self):
        self.last_output_channel = 1024
        self.cfgs = [[3, 1, 16, True, False, 1], [3, 4.5, 24, False, False, (2, 1)], [3, 3.67, 24, False, False, 1], [5, 4, 40, True, True, (2, 1)], [5, 6, 40, True, True, 1], [5, 6, 40, True, True, 1], [5, 3, 48, True, True, 1], [5, 3, 48, True, True, 1], [5, 6, 96, True, True, (2, 1)], [5, 6, 96, True, True, 1], [5, 6, 96, True, True, 1]]

    def initFc(self):
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.pool(x)
        b, c, h, w = x.size()
        assert h == 1, 'the height of conv must be 1'
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        return x


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) ->torch.Tensor:
        return torch.cat(x, self.d)


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, in_planes, class_num, head_w):
        super(AnyHead, self).__init__()
        self.head_w = head_w
        if head_w > 0:
            self.conv = nn.Conv2d(in_planes, head_w, 1, stride=1, padding=0, groups=1, bias=False)
            self.bn = nn.BatchNorm2d(head_w, eps=1e-05, momentum=0.1)
            self.relu = nn.ReLU(inplace=True)
            in_planes = head_w
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, class_num, bias=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x))) if self.head_w > 0 else x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(in_planes, se_planes, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(se_planes, in_planes, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, in_planes, out_planes, stride, bottom_mul, group_width, se_ratio):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(out_planes * bottom_mul))
        g = w_b // group_width
        self.a = nn.Conv2d(in_planes, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-05, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-05, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_ratio:
            se_planes = int(round(in_planes * se_ratio))
            self.se = SE(w_b, se_planes)
        self.c = nn.Conv2d(w_b, out_planes, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, in_planes, out_planes, stride, bottom_mul=1.0, group_width=1, se_ratio=None):
        super(ResBottleneckBlock, self).__init__()
        self.proj_block = in_planes != out_planes or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(in_planes, out_planes, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1)
        self.f = BottleneckTransform(in_planes, out_planes, stride, bottom_mul, group_width, se_ratio)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, in_planes, out_planes):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_planes, out_planes, stride, depth, block_fun, bottom_mul, group_width, se_ratio):
        super(AnyStage, self).__init__()
        for i in range(depth):
            b_stride = stride if i == 0 else 1
            b_w_in = in_planes if i == 0 else out_planes
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fun(b_w_in, out_planes, b_stride, bottom_mul, group_width, se_ratio))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def adjust_block_compatibility(width_per_stage, bot_muls_per_stage, group_w_per_stage):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(width_per_stage) == len(bot_muls_per_stage) == len(group_w_per_stage)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(width_per_stage, bot_muls_per_stage, group_w_per_stage))
    assert all(b < 1 or b % 1 == 0 for b in bot_muls_per_stage)
    vs = [int(max(1, w * b)) for w, b in zip(width_per_stage, bot_muls_per_stage)]
    group_w_per_stage = [int(min(g, v)) for g, v in zip(group_w_per_stage, vs)]
    ms = [(np.lcm(g, int(b)) if b > 1 else g) for g, b in zip(group_w_per_stage, bot_muls_per_stage)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    width_per_stage = [int(v / b) for v, b in zip(vs, bot_muls_per_stage)]
    assert all(w * b % g == 0 for w, b, g in zip(width_per_stage, bot_muls_per_stage, group_w_per_stage))
    return width_per_stage, bot_muls_per_stage, group_w_per_stage


def generate_regnet(anynet_slope, anynet_initial_width, channel_controller, depth, divisor=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert anynet_slope >= 0 and anynet_initial_width > 0 and channel_controller > 1 and anynet_initial_width % divisor == 0
    ws_cont = np.arange(depth) * anynet_slope + anynet_initial_width
    ks = np.round(np.log(ws_cont / anynet_initial_width) / np.log(channel_controller))
    ws_all = anynet_initial_width * np.power(channel_controller, ks)
    ws_all = np.round(np.divide(ws_all, divisor)).astype(int) * divisor
    ws, ds = np.unique(ws_all, return_counts=True)
    num_stages, total_stages = len(ws), ks.max() + 1
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


class RegNet(nn.Module):

    def __init__(self, anynet_slope, anynet_initial_width, channel_controller, depth, group_w, class_num=0, head_w=0):
        super(RegNet, self).__init__()
        stem_w = 32
        se_ratio = 0.25
        width_per_stage, depth_per_stage = generate_regnet(anynet_slope, anynet_initial_width, channel_controller, depth)[0:2]
        group_w_per_stage = [group_w for _ in width_per_stage]
        bot_muls_per_stage = [(1.0) for _ in width_per_stage]
        stride_per_stage = self.auditConfig(width_per_stage)
        width_per_stage, bot_muls_per_stage, group_w_per_stage = adjust_block_compatibility(width_per_stage, bot_muls_per_stage, group_w_per_stage)
        stage_params = list(zip(depth_per_stage, width_per_stage, stride_per_stage, bot_muls_per_stage, group_w_per_stage))
        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (depth, w, s, bottom_mul, group_width) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, depth, ResBottleneckBlock, bottom_mul, group_width, se_ratio))
            prev_w = w
        if class_num > 0:
            self.head = AnyHead(in_planes=prev_w, class_num=class_num, head_w=head_w)
        initWeightsKaiming(self)

    def auditConfig(self, width_per_stage):
        stride_per_stage = [(2) for _ in width_per_stage]
        return stride_per_stage

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class RegNetSmall(RegNet):

    def __init__(self, class_num=0, head_w=0):
        super(RegNetSmall, self).__init__(27.89, 48, 2.09, 16, 8, class_num, head_w)


class RegNetMedium(RegNet):

    def __init__(self, class_num=0, head_w=0):
        super(RegNetMedium, self).__init__(20.71, 48, 2.65, 27, 24, class_num, head_w)


class RegNetLarge(RegNet):

    def __init__(self, class_num=0, head_w=0):
        super(RegNetLarge, self).__init__(31.41, 96, 2.24, 22, 64, class_num, head_w)


class RegNetOCR(RegNet):

    def __init__(self, class_num=0, head_w=0):
        super(RegNetOCR, self).__init__(27.89, 48, 2.09, 16, 8, class_num, head_w)

    def auditConfig(self, width_per_stage):
        stride_per_stage = [(2, 1), (2, 1), (2, 1), 2]
        return stride_per_stage


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        kernel_size = 3
        padding = 1
        padding_11 = padding - kernel_size // 2
        self.relu = nn.ReLU()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = Conv2dBNWithName(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = Conv2dBNWithName(in_channels, out_channels, 1, stride, padding_11, groups)

    def forward(self, inputs):
        if self.deploy:
            return self.relu(self.rbr_reparam(inputs))
        id_out = self.rbr_identity(inputs) if self.rbr_identity else 0
        return self.relu(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor_for_sequential(self, branch):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor_for_bn(self, branch):
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value)
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            return self._fuse_bn_tensor_for_sequential(branch)
        elif isinstance(branch, nn.BatchNorm2d):
            return self._fuse_bn_tensor_for_bn(branch)

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach(), bias.detach()


logger = logging.getLogger('DEEPVAC')


class RepVGG(nn.Module):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.auditConfig()
        self.in_planes = min(64, self.cfgs[0][0])
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, stride=2, deploy=self.deploy)
        self.cur_layer_idx = 1
        layers = []
        for planes, num_blocks, stride in self.cfgs:
            layers.extend(self._make_layer(planes, num_blocks, stride))
        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(self.cfgs[-1][0], class_num)
        initWeightsKaiming(self)

    def auditConfig(self):
        self.cfgs = None
        self.override_groups_map = None
        LOG.logE('You must reimplement auditConfig() to initialize self.cfgs and self.override_groups_map', exit=True)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, stride=stride, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return blocks

    def forward(self, x):
        out = self.stage0(x)
        out = self.layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RepVGGASmall(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGASmall, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[48, 2, 2], [96, 4, 2], [192, 14, 2], [1280, 1, 2]]
        self.override_groups_map = dict()


class RepVGGAMedium(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGAMedium, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[64, 2, 2], [128, 4, 2], [256, 14, 2], [1280, 1, 2]]
        self.override_groups_map = dict()


class RepVGGALarge(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGALarge, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[96, 2, 2], [192, 4, 2], [384, 14, 2], [1408, 1, 2]]
        self.override_groups_map = dict()


class RepVGGBSmall(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBSmall, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[64, 4, 2], [128, 6, 2], [256, 16, 2], [1280, 1, 2]]
        self.override_groups_map = dict()


class RepVGGBMedium(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMedium, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[128, 4, 2], [256, 6, 2], [512, 16, 2], [2048, 1, 2]]
        self.override_groups_map = dict()


class RepVGGBLarge(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLarge, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[160, 4, 2], [320, 6, 2], [640, 16, 2], [2560, 1, 2]]
        self.override_groups_map = dict()


class RepVGGBMediumG2(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMediumG2, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[128, 4, 2], [256, 6, 2], [512, 16, 2], [2048, 1, 2]]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: (2) for l in optional_groupwise_layers}


class RepVGGBMediumG4(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMediumG4, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[128, 4, 2], [256, 6, 2], [512, 16, 2], [2048, 1, 2]]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: (4) for l in optional_groupwise_layers}


class RepVGGBLargeG2(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLargeG2, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[160, 4, 2], [320, 6, 2], [640, 16, 2], [2560, 1, 2]]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: (2) for l in optional_groupwise_layers}


class RepVGGBLargeG4(RepVGG):

    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLargeG4, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [[160, 4, 2], [320, 6, 2], [640, 16, 2], [2560, 1, 2]]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: (4) for l in optional_groupwise_layers}


class ResBnBlock(nn.Module):

    def __init__(self, in_channel, expand_channel, out_channel, kernel_size, stride, shortcut=True):
        super(ResBnBlock, self).__init__()
        self.layer = nn.Sequential(Conv2dBNReLU(in_channel, expand_channel, 1, 1), Conv2dBNReLU(expand_channel, expand_channel, kernel_size, stride, kernel_size // 2, expand_channel), Conv2dBN(expand_channel, out_channel, 1, 1))
        self.shortcut = shortcut and in_channel == out_channel

    def forward(self, x):
        return self.layer(x) + x if self.shortcut else self.layer(x)


class InvertedResidualFacialKpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, expansion_factor: int=6, kernel_size: int=3, stride: int=2, padding: int=1, is_residual: bool=True):
        super(InvertedResidualFacialKpBlock, self).__init__()
        assert stride in [1, 2]
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False), nn.BatchNorm2d(in_channels * expansion_factor), nn.ReLU(inplace=True), nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size, stride, padding, 1, groups=in_channels * expansion_factor, bias=False), nn.BatchNorm2d(in_channels * expansion_factor), nn.ReLU(inplace=True), nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.is_residual = is_residual if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block


class ResnetBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, outplanes: int, stride: int=1):
        super(ResnetBasicBlock, self).__init__()
        self.conv1 = Conv2dBNReLU(in_planes=inplanes, out_planes=outplanes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outplanes))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):

    def __init__(self, class_num: int=1000):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.class_num = class_num
        self.auditConfig()
        self.conv1 = Conv2dBNReLU(in_planes=3, out_planes=self.inplanes, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []
        for outp, layer_num, stride in self.cfgs:
            layers.append(self.block(self.inplanes, outp, stride))
            self.inplanes = outp * self.block.expansion
            for _ in range(1, layer_num):
                layers.append(self.block(self.inplanes, outp))
        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.initFc()
        initWeightsKaiming(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = self.avgpool(x)
        return self.forward_cls(x)

    def forward_cls(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def initFc(self):
        self.fc = nn.Linear(512 * self.block.expansion, self.class_num)

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [[64, 2, 1], [128, 2, 2], [256, 2, 2], [512, 2, 2]]


class ResNet34(ResNet18):

    def __init__(self, class_num: int=1000):
        super(ResNet34, self).__init__(class_num)

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [[64, 3, 1], [128, 4, 2], [256, 6, 2], [512, 3, 2]]


class ResNet50(ResNet18):

    def __init__(self, class_num: int=1000):
        super(ResNet50, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [[64, 3, 1], [128, 4, 2], [256, 6, 2], [512, 3, 2]]


class ResNet101(ResNet18):

    def __init__(self, class_num: int=1000):
        super(ResNet101, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [[64, 3, 1], [128, 4, 2], [256, 23, 2], [512, 3, 2]]


class ResNet152(ResNet18):

    def __init__(self, class_num: int=1000):
        super(ResNet152, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [[64, 3, 1], [128, 8, 2], [256, 36, 2], [512, 3, 2]]


class ResNet18OCR(ResNet18):

    def __init__(self):
        super(ResNet18OCR, self).__init__()

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [[64, 2, 1], [128, 2, (2, 1)], [256, 2, (2, 1)], [512, 2, (2, 1)]]

    def initFc(self):
        self.avgpool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.avgpool(x)
        b, c, h, w = x.size()
        assert h == 1, 'the height of conv must be 1'
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        return x


class SPP(nn.Module):

    def __init__(self, in_planes, out_planes, pool_kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_planes = in_planes // 2
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBNHardswish(hidden_planes * (len(pool_kernel_size) + 1), out_planes, 1, 1)
        self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pool_list], 1))


class SSH(nn.Module):

    def __init__(self, in_planes: int, out_planes: int):
        super(SSH, self).__init__()
        assert out_planes % 4 == 0
        leaky = 0.1 if out_planes <= 64 else 0
        self.conv3X3 = Conv2dBN(in_planes, out_planes // 2, padding=1)
        self.conv5X5_1 = Conv2dBNLeakyReLU(in_planes, out_planes // 4, padding=1, leaky=leaky)
        self.conv5X5_2 = Conv2dBN(out_planes // 4, out_planes // 4, padding=1)
        self.conv7X7_2 = Conv2dBNLeakyReLU(out_planes // 4, out_planes // 4, padding=1, leaky=leaky)
        self.conv7x7_3 = Conv2dBN(out_planes // 4, out_planes // 4, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = self.relu(out)
        return out


def get_fuser_module_index(mod_list):
    rc = []
    if len(mod_list) < 2:
        return rc
    keys = sorted(list(DEFAULT_OP_LIST_TO_FUSER_METHOD.keys()), key=lambda x: len(x), reverse=True)
    mod2fused_list = [list(x) for x in keys]
    for mod2fused in mod2fused_list:
        if len(mod2fused) > len(mod_list):
            continue
        mod2fused_idx = [(i, i + len(mod2fused)) for i in range(len(mod_list) - len(mod2fused) + 1) if mod_list[i:i + len(mod2fused)] == mod2fused]
        if not mod2fused_idx:
            continue
        for idx in mod2fused_idx:
            start, end = idx
            mod_list[start:end] = [None] * len(mod2fused)
        rc.extend(mod2fused_idx)
    return rc


def auto_fuse_model(model):
    module_names = []
    module_types = []
    for name, m in model.named_modules():
        module_names.append(name)
        module_types.append(type(m))
    if len(module_types) < 2:
        return model
    module_idxs = get_fuser_module_index(module_types)
    modules_to_fuse = [module_names[mi[0]:mi[1]] for mi in module_idxs]
    new_model = torch.quantization.fuse_modules(model, modules_to_fuse)
    return new_model


class DeepvacQAT(torch.nn.Module):

    def __init__(self, net2qat):
        super(DeepvacQAT, self).__init__()
        self.quant = QuantStub()
        self.net2qat = auto_fuse_model(net2qat)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.net2qat(x)
        x = self.dequant(x)
        return x


class ArcFace(nn.Module):

    def __init__(self, embedding_size, class_num, s=32.0, m=0.5, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_feature = embedding_size
        self.out_feature = class_num
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.Tensor(self.out_feature, self.in_feature))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.kernel))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine - self.th > 0, phi, cosine - self.mm) if not self.easy_margin else torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s
        return output


class CurricularFace(nn.Module):

    def __init__(self, embedding_size, class_num, world_size=1, s=64.0, m=0.5):
        super(CurricularFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.world_size = world_size
        self.kernel = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, x, label):
        kernel_norm = F.normalize(self.kernel)
        cos_theta = torch.mm(x, kernel_norm).clamp(-1, 1)
        target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            if self.world_size != 1:
                dist.all_reduce(self.t, dist.ReduceOp.SUM)
            self.t = self.t / self.world_size
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output


class AttrDict(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, memo=None):
        try:
            ad = AttrDict(copy.deepcopy(dict(self), memo=memo))
        except Exception as e:
            None
            ad = AttrDict()
        return ad

    def clone(self):
        return copy.deepcopy(self)


class LossBase(nn.Module):

    def __init__(self, deepvac_config):
        super(LossBase, self).__init__()
        self.deepvac_loss_config = deepvac_config.loss
        self.initConfig()
        self.auditConfig()

    def initConfig(self):
        if self.name() not in self.deepvac_loss_config.keys():
            self.deepvac_loss_config[self.name()] = AttrDict()
        self.config = self.deepvac_loss_config[self.name()]

    def addUserConfig(self, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
        module_name = 'config.loss.{}'.format(self.name())
        return addUserConfig(module_name, config_name, user_give, developer_give, is_user_mandatory)

    def name(self):
        return self.__class__.__name__

    def auditConfig(self):
        raise Exception('Not implemented!')


class MaskL1Loss(LossBase):

    def __init__(self, deepvac_config):
        super(MaskL1Loss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.eps = self.addUserConfig('eps', self.config.eps, 1e-06)

    def __call__(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class DiceLoss(LossBase):

    def __init__(self, deepvac_config):
        super(DiceLoss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.eps = self.addUserConfig('eps', self.config.eps, 1e-06)

    def __call__(self, pred: torch.Tensor, gt, mask, weights=None):
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class BalanceCrossEntropyLoss(LossBase):

    def __init__(self, deepvac_config):
        super(BalanceCrossEntropyLoss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.negative_ratio = self.addUserConfig('negative_ratio', self.config.negative_ratio, 3.0)
        self.eps = self.addUserConfig('eps', self.config.eps, 1e-06)

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, return_origin=False):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss


class BCEBlurWithLogitsLoss(LossBase):

    def __init__(self, deepvac_config):
        super(BCEBlurWithLogitsLoss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.alpha = self.addUserConfig('alpha', self.config.alpha, 0.05)
        self.reduction = self.addUserConfig('reduction', self.config.reduction, 'none')
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(LossBase):

    def __init__(self, deepvac_config):
        super(FocalLoss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.alpha = self.addUserConfig('alpha', self.config.alpha, 0.25)
        self.gamma = self.addUserConfig('gamma', self.config.gamma, 1.5)
        self.reduction = self.addUserConfig('reduction', self.config.reduction, 'none')
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(LossBase):

    def __init__(self, deepvac_config):
        super(QFocalLoss, self).__init__(deepvac_config)

    def auditConfig(self):
        self.alpha = self.addUserConfig('alpha', self.config.alpha, 0.25)
        self.gamma = self.addUserConfig('gamma', self.config.gamma, 1.5)
        self.reduction = self.addUserConfig('reduction', self.config.reduction, 'none')
        self.loss_fcn = nn.BCEWithLogitsLoss()

    def __call__(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WingLoss(nn.Module):

    def __init__(self):
        super(WingLoss, self).__init__()

    def forward(self, pred, truth, w=10.0, epsilon=2.0):
        x = truth - pred
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = torch.abs(x)
        losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x / epsilon), absolute_x - c)
        return torch.sum(losses) / (len(losses) * 1.0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AnyHead,
     lambda: ([], {'in_planes': 4, 'class_num': 4, 'head_w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnyStage,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1, 'depth': 1, 'block_fun': _mock_layer, 'bottom_mul': 4, 'group_width': 4, 'se_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BNPReLU,
     lambda: ([], {'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckIR,
     lambda: ([], {'inplanes': 4, 'outplanes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckTransform,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1, 'bottom_mul': 4, 'group_width': 4, 'se_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBN,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBNHardswish,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBNHswish,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBNLeakyReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dBNPReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBNWithName,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dBnAct,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dDilatedBN,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthWiseConv2d,
     lambda: ([], {'inplanes': 4, 'outplanes': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 4, 'use_se': 4, 'use_hs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidualFacialKpBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MobileNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MobileNetV3Large,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RegNetLarge,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RegNetMedium,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RegNetOCR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RegNetSmall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGALarge,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGAMedium,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGASmall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBLarge,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBLargeG2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBLargeG4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBMedium,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBMediumG2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBMediumG4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBSmall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RepVGGBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBottleneckBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet101,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet152,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResnetBasicBlock,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SE,
     lambda: ([], {'in_planes': 4, 'se_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPP,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSH,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleStemIN,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DeepVAC_deepvac(_paritybench_base):
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

