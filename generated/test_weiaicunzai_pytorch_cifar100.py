import sys
_module = sys.modules[__name__]
del sys
conf = _module
global_settings = _module
dataset = _module
lr_finder = _module
attention = _module
densenet = _module
googlenet = _module
inceptionv3 = _module
inceptionv4 = _module
mobilenet = _module
mobilenetv2 = _module
nasnet = _module
preactresnet = _module
resnet = _module
resnext = _module
rir = _module
senet = _module
shufflenet = _module
shufflenetv2 = _module
squeezenet = _module
vgg = _module
xception = _module
test = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


import numpy as np


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn.functional as F


import math


from functools import partial


from torch.autograd import Variable


class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, downsample the featuremap
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = int(out_channels / 4)
        self.residual_function = nn.Sequential(nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True), nn.Conv2d(in_channels,
            bottleneck_channels, 1, stride), nn.BatchNorm2d(
            bottleneck_channels), nn.ReLU(inplace=True), nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3, padding=1), nn.
            BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True), nn.
            Conv2d(bottleneck_channels, out_channels, 1))
        self.shortcut = nn.Sequential()
        if stride != 2 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=
                stride)

    def forward(self, x):
        res = self.residual_function(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class AttentionModule1(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown4 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r)
        self.shortcut_short = PreActResidualUnit(in_channels, out_channels, 1)
        self.shortcut_long = PreActResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=
            True), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn
            .Sigmoid())
        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = x.size(2), x.size(3)
        x_t = self.trunk(x)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)
        shape1 = x_s.size(2), x_s.size(3)
        shortcut_long = self.shortcut_long(x_s)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)
        shape2 = x_s.size(2), x_s.size(3)
        shortcut_short = self.soft_resdown3(x_s)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)
        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)
        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=
            True), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn
            .Sigmoid())
        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = x.size(2), x.size(3)
        x_t = self.trunk(x)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)
        shape1 = x_s.size(2), x_s.size(3)
        shortcut = self.shortcut(x_s)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)
        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)
        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels
        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)
        self.sigmoid = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=
            True), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn
            .Sigmoid())
        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = x.size(2), x.size(3)
        x_t = self.trunk(x)
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resup1(x_s)
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)
        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)
        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """

    def __init__(self, block_num, class_num=100):
        super().__init__()
        self.pre_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
            stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(64, 256, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(256, 512, block_num[1], AttentionModule2
            )
        self.stage3 = self._make_stage(512, 1024, block_num[2],
            AttentionModule3)
        self.stage4 = nn.Sequential(PreActResidualUnit(1024, 2048, 2),
            PreActResidualUnit(2048, 2048, 1), PreActResidualUnit(2048, 
            2048, 1))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_stage(self, in_channels, out_channels, num, block):
        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2))
        for _ in range(num):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bottle_neck = nn.Sequential(nn.BatchNorm2d(in_channels), nn.
            ReLU(inplace=True), nn.Conv2d(in_channels, inner_channel,
            kernel_size=1, bias=False), nn.BatchNorm2d(inner_channel), nn.
            ReLU(inplace=True), nn.Conv2d(inner_channel, growth_rate,
            kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(nn.BatchNorm2d(in_channels), nn.
            Conv2d(in_channels, out_channels, 1, bias=False), nn.AvgPool2d(
            2, stride=2))

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5,
        num_class=100):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1,
            bias=False)
        self.features = nn.Sequential()
        for index in range(len(nblocks) - 1):
            self.features.add_module('dense_block_layer_{}'.format(index),
                self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(index),
                Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.features.add_module('dense_block{}'.format(len(nblocks) - 1),
            self._make_dense_layers(block, inner_channels, nblocks[len(
            nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index),
                block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


class Inception(nn.Module):

    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce,
        n5x5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channels, n1x1, kernel_size
            =1), nn.BatchNorm2d(n1x1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(input_channels, n3x3_reduce,
            kernel_size=1), nn.BatchNorm2d(n3x3_reduce), nn.ReLU(inplace=
            True), nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(input_channels, n5x5_reduce,
            kernel_size=1), nn.BatchNorm2d(n5x5_reduce), nn.ReLU(inplace=
            True), nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5), nn.ReLU(inplace=True), nn.Conv2d(
            n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn
            .ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.
            Conv2d(input_channels, pool_proj, kernel_size=1), nn.
            BatchNorm2d(pool_proj), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)],
            dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3,
            padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)
        output = self.maxpool(output)
        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)
        output = self.maxpool(output)
        output = self.a5(output)
        output = self.b5(output)
        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False,
            **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(BasicConv2d(input_channels, 48,
            kernel_size=1), BasicConv2d(48, 64, kernel_size=5, padding=2))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 64,
            kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1))
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=
            1, padding=1), BasicConv2d(input_channels, pool_features,
            kernel_size=3, padding=1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branchpool = self.branchpool(x)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=3,
            stride=2)
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 64,
            kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2))
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3stack = self.branch3x3stack(x)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, c7,
            kernel_size=1), BasicConv2d(c7, c7, kernel_size=(7, 1), padding
            =(3, 0)), BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.branch7x7stack = nn.Sequential(BasicConv2d(input_channels, c7,
            kernel_size=1), BasicConv2d(c7, c7, kernel_size=(7, 1), padding
            =(3, 0)), BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3
            )), BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride
            =1, padding=1), BasicConv2d(input_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7stack = self.branch7x7stack(x)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 320, kernel_size=3, stride=2))
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 192, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(192, 192, kernel_size=(7, 1),
            padding=(3, 0)), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch7x7, branchpool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride
            =1, padding=1), BasicConv2d(input_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [self.branch3x3stack_3a(branch3x3stack), self.
            branch3x3stack_3b(branch3x3stack)]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False,
            **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_Stem(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(BasicConv2d(input_channels, 32,
            kernel_size=3), BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1))
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1,
            padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_conv(x), self.branch3x3_pool(x)]
        x = torch.cat(x, 1)
        x = [self.branch7x7a(x), self.branch7x7b(x)]
        x = torch.cat(x, 1)
        x = [self.branchpoola(x), self.branchpoolb(x)]
        x = torch.cat(x, 1)
        return x


class InceptionA(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 64,
            kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 64,
            kernel_size=1), BasicConv2d(64, 96, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=
            1, padding=1), BasicConv2d(input_channels, 96, kernel_size=1))

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branch1x1(x),
            self.branchpool(x)]
        return torch.cat(x, 1)


class ReductionA(nn.Module):

    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k,
            kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2
            )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7stack = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 192, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(192, 224, kernel_size=(7, 1),
            padding=(3, 0)), BasicConv2d(224, 224, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            padding=(3, 0)))
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1
            ), BasicConv2d(input_channels, 128, kernel_size=1))

    def forward(self, x):
        x = [self.branch1x1(x), self.branch7x7(x), self.branch7x7stack(x),
            self.branchpool(x)]
        return torch.cat(x, 1)


class ReductionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 256,
            kernel_size=1), BasicConv2d(256, 256, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1),
            padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2,
            padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 192, kernel_size=3, stride=2,
            padding=1))
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [self.branch3x3(x), self.branch7x7(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionC(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 384,
            kernel_size=1), BasicConv2d(384, 448, kernel_size=(1, 3),
            padding=(0, 1)), BasicConv2d(448, 512, kernel_size=(3, 1),
            padding=(1, 0)))
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding
            =(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding
            =(0, 1))
        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=
            1, padding=1), BasicConv2d(input_channels, 256, kernel_size=1))

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [self.branch3x3stacka(branch3x3stack_output
            ), self.branch3x3stackb(branch3x3stack_output)]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)
        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [self.branch3x3a(branch3x3_output), self.
            branch3x3b(branch3x3_output)]
        branch3x3_output = torch.cat(branch3x3_output, 1)
        branch1x1_output = self.branch1x1(x)
        branchpool = self.branchpool(x)
        output = [branch1x1_output, branch3x3_output, branch3x3stack_output,
            branchpool]
        return torch.cat(output, 1)


class InceptionV4(nn.Module):

    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=100):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_a = self._generate_inception_module(384, 384, A,
            InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels,
            1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C,
            InceptionC)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)
        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels,
        block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module('{}_{}'.format(block.__name__, l), block(
                input_channels))
            input_channels = output_channels
        return layers


class InceptionResNetA(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 32,
            kernel_size=1), BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1))
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 32,
            kernel_size=1), BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x), self.
            branch3x3stack(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)
        output = self.bn(shortcut + residual)
        output = self.relu(output)
        return output


class InceptionResNetB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(BasicConv2d(input_channels, 128,
            kernel_size=1), BasicConv2d(128, 160, kernel_size=(1, 7),
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            padding=(3, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 1154, kernel_size=1)
        self.bn = nn.BatchNorm2d(1154)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch7x7(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shortcut = self.shortcut(x)
        output = self.bn(residual + shortcut)
        output = self.relu(output)
        return output


class InceptionResNetC(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 192,
            kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 3),
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            padding=(1, 0)))
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 2048, kernel_size=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shorcut = self.shorcut(x)
        output = self.bn(shorcut + residual)
        output = self.relu(output)
        return output


class InceptionResNetReductionA(nn.Module):

    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, k,
            kernel_size=1), BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2))
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2
            )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [self.branch3x3stack(x), self.branch3x3(x), self.branchpool(x)]
        return torch.cat(x, 1)


class InceptionResNetReductionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branchpool = nn.MaxPool2d(3, stride=2)
        self.branch3x3a = nn.Sequential(BasicConv2d(input_channels, 256,
            kernel_size=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch3x3b = nn.Sequential(BasicConv2d(input_channels, 256,
            kernel_size=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch3x3stack = nn.Sequential(BasicConv2d(input_channels, 256,
            kernel_size=1), BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2))

    def forward(self, x):
        x = [self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x),
            self.branchpool(x)]
        x = torch.cat(x, 1)
        return x


class InceptionResNetV2(nn.Module):

    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, class_nums=100):
        super().__init__()
        self.stem = Inception_Stem(3)
        self.inception_resnet_a = self._generate_inception_module(384, 384,
            A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(
            output_channels, 1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 
            2048, C, InceptionResNetC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(2048, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x = self.linear(x)
        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels,
        block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module('{}_{}'.format(block.__name__, l), block(
                input_channels))
            input_channels = output_channels
        return layers


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(input_channels,
            input_channels, kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True))
        self.pointwise = nn.Sequential(nn.Conv2d(input_channels,
            output_channels, 1), nn.BatchNorm2d(output_channels), nn.ReLU(
            inplace=True))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
            **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin 
                         a network uniformly at each layer. For a given 
                         layer and width multiplier α, the number of 
                         input channels M becomes αM and the number of 
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
        super().__init__()
        alpha = width_multiplier
        self.stem = nn.Sequential(BasicConv2d(3, int(32 * alpha), 3,
            padding=1, bias=False), DepthSeperabelConv2d(int(32 * alpha),
            int(64 * alpha), 3, padding=1, bias=False))
        self.conv1 = nn.Sequential(DepthSeperabelConv2d(int(64 * alpha),
            int(128 * alpha), 3, stride=2, padding=1, bias=False),
            DepthSeperabelConv2d(int(128 * alpha), int(128 * alpha), 3,
            padding=1, bias=False))
        self.conv2 = nn.Sequential(DepthSeperabelConv2d(int(128 * alpha),
            int(256 * alpha), 3, stride=2, padding=1, bias=False),
            DepthSeperabelConv2d(int(256 * alpha), int(256 * alpha), 3,
            padding=1, bias=False))
        self.conv3 = nn.Sequential(DepthSeperabelConv2d(int(256 * alpha),
            int(512 * alpha), 3, stride=2, padding=1, bias=False),
            DepthSeperabelConv2d(int(512 * alpha), int(512 * alpha), 3,
            padding=1, bias=False), DepthSeperabelConv2d(int(512 * alpha),
            int(512 * alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512 * alpha), int(512 * alpha), 3,
            padding=1, bias=False), DepthSeperabelConv2d(int(512 * alpha),
            int(512 * alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512 * alpha), int(512 * alpha), 3,
            padding=1, bias=False))
        self.conv4 = nn.Sequential(DepthSeperabelConv2d(int(512 * alpha),
            int(1024 * alpha), 3, stride=2, padding=1, bias=False),
            DepthSeperabelConv2d(int(1024 * alpha), int(1024 * alpha), 3,
            padding=1, bias=False))
        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels, in_channels *
            t, 1), nn.BatchNorm2d(in_channels * t), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride,
            padding=1, groups=in_channels * t), nn.BatchNorm2d(in_channels *
            t), nn.ReLU6(inplace=True), nn.Conv2d(in_channels * t,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual


class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 32, 1, padding=1), nn.
            BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        self.conv1 = nn.Sequential(nn.Conv2d(320, 1280, 1), nn.BatchNorm2d(
            1280), nn.ReLU6(inplace=True))
        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nn.Sequential(*layers)


class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels,
            kernel_size, groups=input_channels, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeperableBranch(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__()
        self.block1 = nn.Sequential(nn.ReLU(), SeperableConv2d(
            input_channels, output_channels, kernel_size, **kwargs), nn.
            BatchNorm2d(output_channels))
        self.block2 = nn.Sequential(nn.ReLU(), SeperableConv2d(
            output_channels, output_channels, kernel_size, stride=1,
            padding=int(kernel_size / 2)), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Fit(nn.Module):
    """Make the cell outputs compatible

    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()
        self.p1 = nn.Sequential(nn.AvgPool2d(1, stride=2), nn.Conv2d(
            prev_filters, int(filters / 2), 1))
        self.p2 = nn.Sequential(nn.ConstantPad2d((0, 1, 0, 1), 0), nn.
            ConstantPad2d((-1, 0, -1, 0), 0), nn.AvgPool2d(1, stride=2), nn
            .Conv2d(prev_filters, int(filters / 2), 1))
        self.bn = nn.BatchNorm2d(filters)
        self.dim_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(prev_filters,
            filters, 1), nn.BatchNorm2d(filters))
        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)
        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)
        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()
        self.dem_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(x_in,
            output_channels, 1, bias=False), nn.BatchNorm2d(output_channels))
        self.block1_left = SeperableBranch(output_channels, output_channels,
            kernel_size=3, padding=1, bias=False)
        self.block1_right = nn.Sequential()
        self.block2_left = SeperableBranch(output_channels, output_channels,
            kernel_size=3, padding=1, bias=False)
        self.block2_right = SeperableBranch(output_channels,
            output_channels, kernel_size=5, padding=2, bias=False)
        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()
        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)
        self.block5_left = SeperableBranch(output_channels, output_channels,
            kernel_size=5, padding=2, bias=False)
        self.block5_right = SeperableBranch(output_channels,
            output_channels, kernel_size=3, padding=1, bias=False)
        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dem_reduce(x)
        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)
        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x


class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()
        self.dim_reduce = nn.Sequential(nn.ReLU(), nn.Conv2d(x_in,
            output_channels, 1), nn.BatchNorm2d(output_channels))
        self.layer1block1_left = SeperableBranch(output_channels,
            output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels,
            output_channels, 5, stride=2, padding=2)
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels,
            output_channels, 7, stride=2, padding=3)
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels,
            output_channels, 5, stride=2, padding=2)
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels,
            output_channels, 3, stride=1, padding=1)
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()
        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dim_reduce(x)
        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h
            )
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev
            )
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev
            )
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(
            layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1
            ) + self.layer2block2_right(layer1block2)
        return torch.cat([layer1block2, layer1block3, layer2block1,
            layer2block2], 1), x


class NasNetA(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter,
        class_num=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stemfilter, 3, padding=1,
            bias=False), nn.BatchNorm2d(stemfilter))
        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters
        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 6, class_num)

    def _make_normal(self, block, repeat, output):
        """make normal cell
        Args:
            block: cell type
            repeat: number of repeated normal cell
            output: output filters for each branch in normal cell
        Returns:
            stacked normal cells
        """
        layers = []
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * 6
        return layers

    def _make_reduction(self, block, output):
        """make normal cell
        Args:
            block: cell type
            output: output filters for each branch in reduction cell
        Returns:
            reduction cell
        """
        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4
        return reduction

    def _make_layers(self, repeat_cell_num, reduction_num):
        layers = []
        for i in range(reduction_num):
            layers.extend(self._make_normal(NormalCell, repeat_cell_num,
                self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))
        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.
            filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreActBasic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size
            =3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn
            .ReLU(inplace=True), nn.Conv2d(out_channels, out_channels *
            PreActBasic.expansion, kernel_size=3, padding=1))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels *
                PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class PreActBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels, out_channels, 1, stride=
            stride), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.
            BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(
            out_channels, out_channels * PreActBottleNeck.expansion, 1))
        self.shortcut = nn.Sequential()
        if (stride != 1 or in_channels != out_channels * PreActBottleNeck.
            expansion):
            self.shortcut = nn.Conv2d(in_channels, out_channels *
                PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class PreActResNet(nn.Module):

    def __init__(self, block, num_block, class_num=100):
        super().__init__()
        self.input_channels = 64
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.
            BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = self._make_layers(block, num_block[0], 64, 1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)
        self.linear = nn.Linear(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []
        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion
        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(nn.Conv2d(in_channels,
            out_channels, kernel_size=3, stride=stride, padding=1, bias=
            False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn
            .Conv2d(out_channels, out_channels * BasicBlock.expansion,
            kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(
            out_channels * BasicBlock.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, 
                out_channels * BasicBlock.expansion, kernel_size=1, stride=
                stride, bias=False), nn.BatchNorm2d(out_channels *
                BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.
            shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(nn.Conv2d(in_channels,
            out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,
            out_channels, stride=stride, kernel_size=3, padding=1, bias=
            False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn
            .Conv2d(out_channels, out_channels * BottleNeck.expansion,
            kernel_size=1, bias=False), nn.BatchNorm2d(out_channels *
            BottleNeck.expansion))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, 
                out_channels * BottleNeck.expansion, stride=stride,
                kernel_size=1, bias=False), nn.BatchNorm2d(out_channels *
                BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.
            shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=
            1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


DEPTH = 4


BASEWIDTH = 64


CARDINALITY = 32


class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        C = CARDINALITY
        D = int(DEPTH * out_channels / BASEWIDTH)
        self.split_transforms = nn.Sequential(nn.Conv2d(in_channels, C * D,
            kernel_size=1, groups=C, bias=False), nn.BatchNorm2d(C * D), nn
            .ReLU(inplace=True), nn.Conv2d(C * D, C * D, kernel_size=3,
            stride=stride, groups=C, padding=1, bias=False), nn.BatchNorm2d
            (C * D), nn.ReLU(inplace=True), nn.Conv2d(C * D, out_channels *
            4, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels * 4))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, 
                out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))


class ResNext(nn.Module):

    def __init__(self, block, num_blocks, class_names=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride
        
        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)


class ResnetInit(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.residual_stream_conv = nn.Conv2d(in_channel, out_channel, 3,
            padding=1, stride=stride)
        self.transient_stream_conv = nn.Conv2d(in_channel, out_channel, 3,
            padding=1, stride=stride)
        self.residual_stream_conv_across = nn.Conv2d(in_channel,
            out_channel, 3, padding=1, stride=stride)
        self.transient_stream_conv_across = nn.Conv2d(in_channel,
            out_channel, 3, padding=1, stride=stride)
        self.residual_bn_relu = nn.Sequential(nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.transient_bn_relu = nn.Sequential(nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.short_cut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=1, stride=stride))

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual_stream_conv(x_residual)
        residual_r_t = self.residual_stream_conv_across(x_residual)
        residual_shortcut = self.short_cut(x_residual)
        transient_t_t = self.transient_stream_conv(x_transient)
        transient_t_r = self.transient_stream_conv_across(x_transient)
        x_residual = self.residual_bn_relu(residual_r_r + transient_t_r +
            residual_shortcut)
        x_transient = self.transient_bn_relu(residual_r_t + transient_t_t)
        return x_residual, x_transient


class RiRBlock(nn.Module):

    def __init__(self, in_channel, out_channel, layer_num, stride, layer=
        ResnetInit):
        super().__init__()
        self.resnetinit = self._make_layers(in_channel, out_channel,
            layer_num, stride)

    def forward(self, x):
        x_residual, x_transient = self.resnetinit(x)
        return x_residual, x_transient

    def _make_layers(self, in_channel, out_channel, layer_num, stride,
        layer=ResnetInit):
        strides = [stride] + [1] * (layer_num - 1)
        layers = nn.Sequential()
        for index, s in enumerate(strides):
            layers.add_module('generalized layers{}'.format(index), layer(
                in_channel, out_channel, s))
            in_channel = out_channel
        return layers


class ResnetInResneet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        base = int(96 / 2)
        self.residual_pre_conv = nn.Sequential(nn.Conv2d(3, base, 3,
            padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.transient_pre_conv = nn.Sequential(nn.Conv2d(3, base, 3,
            padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.rir1 = RiRBlock(base, base, 2, 1)
        self.rir2 = RiRBlock(base, base, 2, 1)
        self.rir3 = RiRBlock(base, base * 2, 2, 2)
        self.rir4 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir5 = RiRBlock(base * 2, base * 2, 2, 1)
        self.rir6 = RiRBlock(base * 2, base * 4, 2, 2)
        self.rir7 = RiRBlock(base * 4, base * 4, 2, 1)
        self.rir8 = RiRBlock(base * 4, base * 4, 2, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(384, num_classes, kernel_size=
            3, stride=2), nn.BatchNorm2d(num_classes), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(900, 450), nn.ReLU(), nn.
            Dropout(), nn.Linear(450, 100))
        self._weight_init()

    def forward(self, x):
        x_residual = self.residual_pre_conv(x)
        x_transient = self.transient_pre_conv(x)
        x_residual, x_transient = self.rir1((x_residual, x_transient))
        x_residual, x_transient = self.rir2((x_residual, x_transient))
        x_residual, x_transient = self.rir3((x_residual, x_transient))
        x_residual, x_transient = self.rir4((x_residual, x_transient))
        x_residual, x_transient = self.rir5((x_residual, x_transient))
        x_residual, x_transient = self.rir6((x_residual, x_transient))
        x_residual, x_transient = self.rir7((x_residual, x_transient))
        x_residual, x_transient = self.rir8((x_residual, x_transient))
        h = torch.cat([x_residual, x_transient], 1)
        h = self.conv1(h)
        h = h.view(h.size()[0], -1)
        h = self.classifier(h)
        return h

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.fill_(0.01)


class BasicResidualSEBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
            3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True), nn.Conv2d(out_channels, out_channels * self
            .expansion, 3, padding=1), nn.BatchNorm2d(out_channels * self.
            expansion), nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, 
                out_channels * self.expansion, 1, stride=stride), nn.
                BatchNorm2d(out_channels * self.expansion))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(out_channels * self.
            expansion, out_channels * self.expansion // r), nn.ReLU(inplace
            =True), nn.Linear(out_channels * self.expansion // r, 
            out_channels * self.expansion), nn.Sigmoid())

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        x = residual * excitation.expand_as(residual) + shortcut
        return F.relu(x)


class BottleneckResidualSEBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
            1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.
            Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(
            out_channels, out_channels * self.expansion, 1), nn.BatchNorm2d
            (out_channels * self.expansion), nn.ReLU(inplace=True))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(out_channels * self.
            expansion, out_channels * self.expansion // r), nn.ReLU(inplace
            =True), nn.Linear(out_channels * self.expansion // r, 
            out_channels * self.expansion), nn.Sigmoid())
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, 
                out_channels * self.expansion, 1, stride=stride), nn.
                BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        x = residual * excitation.expand_as(residual) + shortcut
        return F.relu(x)


class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        super().__init__()
        self.in_channels = 64
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.
            BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 516, 2)
        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_stage(self, block, num, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1
        return nn.Sequential(*layers)


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
            **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(input_channels,
            output_channels, kernel_size, **kwargs), nn.BatchNorm2d(
            output_channels))

    def forward(self, x):
        return self.depthwise(x)


class PointwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(nn.Conv2d(input_channels,
            output_channels, 1, **kwargs), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        return self.pointwise(x)


class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()
        self.bottlneck = nn.Sequential(PointwiseConv2d(input_channels, int(
            output_channels / 4), groups=groups), nn.ReLU(inplace=True))
        if stage == 2:
            self.bottlneck = nn.Sequential(PointwiseConv2d(input_channels,
                int(output_channels / 4), groups=groups), nn.ReLU(inplace=True)
                )
        self.channel_shuffle = ChannelShuffle(groups)
        self.depthwise = DepthwiseConv2d(int(output_channels / 4), int(
            output_channels / 4), 3, groups=int(output_channels / 4),
            stride=stride, padding=1)
        self.expand = PointwiseConv2d(int(output_channels / 4),
            output_channels, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)
            self.expand = PointwiseConv2d(int(output_channels / 4), 
                output_channels - input_channels, groups=groups)
            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)
        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)
        return output


class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, num_classes=100, groups=3):
        super().__init__()
        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]
        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]
        self.stage2 = self._make_stage(ShuffleNetUnit, num_blocks[0],
            out_channels[1], stride=2, stage=2, groups=groups)
        self.stage3 = self._make_stage(ShuffleNetUnit, num_blocks[1],
            out_channels[2], stride=2, stage=3, groups=groups)
        self.stage4 = self._make_stage(ShuffleNetUnit, num_blocks[2],
            out_channels[3], stride=2, stage=4, groups=groups)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, stage,
        groups):
        """make shufflenet stage 

        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution 
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)
        stage = []
        for stride in strides:
            stage.append(block(self.input_channels, output_channels, stride
                =stride, stage=stage, groups=groups))
            self.input_channels = output_channels
        return nn.Sequential(*stage)


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """
    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(
                inplace=True), nn.Conv2d(in_channels, in_channels, 3,
                stride=stride, padding=1, groups=in_channels), nn.
                BatchNorm2d(in_channels), nn.Conv2d(in_channels, int(
                out_channels / 2), 1), nn.BatchNorm2d(int(out_channels / 2)
                ), nn.ReLU(inplace=True))
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                in_channels, 3, stride=stride, padding=1, groups=
                in_channels), nn.BatchNorm2d(in_channels), nn.Conv2d(
                in_channels, int(out_channels / 2), 1), nn.BatchNorm2d(int(
                out_channels / 2)), nn.ReLU(inplace=True))
        else:
            self.shortcut = nn.Sequential()
            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(
                inplace=True), nn.Conv2d(in_channels, in_channels, 3,
                stride=stride, padding=1, groups=in_channels), nn.
                BatchNorm2d(in_channels), nn.Conv2d(in_channels,
                in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(
                inplace=True))

    def forward(self, x):
        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x
        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)
        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, class_num=100):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')
        self.pre = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.
            BatchNorm2d(24))
        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(nn.Conv2d(out_channels[2], out_channels[
            3], 1), nn.BatchNorm2d(out_channels[3]), nn.ReLU(inplace=True))
        self.fc = nn.Linear(out_channels[3], class_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))
        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1
        return nn.Sequential(*layers)


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channel, squzee_channel, 
            1), nn.BatchNorm2d(squzee_channel), nn.ReLU(inplace=True))
        self.expand_1x1 = nn.Sequential(nn.Conv2d(squzee_channel, int(
            out_channel / 2), 1), nn.BatchNorm2d(int(out_channel / 2)), nn.
            ReLU(inplace=True))
        self.expand_3x3 = nn.Sequential(nn.Conv2d(squzee_channel, int(
            out_channel / 2), 3, padding=1), nn.BatchNorm2d(int(out_channel /
            2)), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([self.expand_1x1(x), self.expand_3x3(x)], 1)
        return x


class SqueezeNet(nn.Module):
    """mobile net with simple bypass"""

    def __init__(self, class_num=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 96, 3, padding=1), nn.
            BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)
        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)
        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)
        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)
        f9 = self.fire9(f8)
        c10 = self.conv10(f9)
        x = self.avg(c10)
        x = x.view(x.size(0), -1)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512, 4096), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Linear(4096, num_class))

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels,
            kernel_size, groups=input_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias
            =False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EntryFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=
            False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=
            False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3_residual = nn.Sequential(SeperableConv2d(64, 128, 3,
            padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.conv3_shortcut = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128))
        self.conv4_residual = nn.Sequential(nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.MaxPool2d(3, stride=2, padding=1))
        self.conv4_shortcut = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2
            ), nn.BatchNorm2d(256))
        self.conv5_residual = nn.Sequential(nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1), nn.BatchNorm2d(728),
            nn.ReLU(inplace=True), SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728), nn.MaxPool2d(3, 1, padding=1))
        self.conv5_shortcut = nn.Sequential(nn.Conv2d(256, 728, 1), nn.
            BatchNorm2d(728))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut
        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(
            728, 728, 3, padding=1), nn.BatchNorm2d(728))
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(
            728, 728, 3, padding=1), nn.BatchNorm2d(728))
        self.conv3 = nn.Sequential(nn.ReLU(inplace=True), SeperableConv2d(
            728, 728, 3, padding=1), nn.BatchNorm2d(728))

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        shortcut = self.shortcut(x)
        return shortcut + residual


class MiddleFlow(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())
        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.ReLU(), SeperableConv2d(728, 728, 
            3, padding=1), nn.BatchNorm2d(728), nn.ReLU(), SeperableConv2d(
            728, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.MaxPool2d(3,
            stride=2, padding=1))
        self.shortcut = nn.Sequential(nn.Conv2d(728, 1024, 1, stride=2), nn
            .BatchNorm2d(1024))
        self.conv = nn.Sequential(SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536), nn.ReLU(inplace=True), SeperableConv2d(
            1536, 2048, 3, padding=1), nn.BatchNorm2d(2048), nn.ReLU(
            inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)
        return output


class Xception(nn.Module):

    def __init__(self, block, num_class=100):
        super().__init__()
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

