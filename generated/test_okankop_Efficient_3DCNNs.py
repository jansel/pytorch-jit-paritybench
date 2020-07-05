import sys
_module = sys.modules[__name__]
del sys
calculate_FLOP = _module
dataset = _module
jester = _module
kinetics = _module
ucf101 = _module
main = _module
mean = _module
model = _module
c3d = _module
mobilenet = _module
mobilenetv2 = _module
resnet = _module
resnext = _module
shufflenet = _module
shufflenetv2 = _module
squeezenet = _module
opts = _module
spatial_transforms = _module
speed_gpu = _module
target_transforms = _module
temporal_transforms = _module
test = _module
test_models = _module
thop = _module
count_hooks = _module
utils = _module
train = _module
eval_kinetics = _module
eval_ucf101 = _module
jester_json = _module
kinetics_json = _module
n_frames_jester = _module
n_frames_kinetics = _module
n_frames_ucf101_hmdb51 = _module
ucf101_json = _module
video_accuracy = _module
video_jpg = _module
video_jpg_kinetics = _module
video_jpg_ucf101_hmdb51 = _module
validation = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import math


import torch


import torch.nn.init as init


import torch.nn.functional as F


from torch.autograd import Variable


from functools import partial


from collections import OrderedDict


from torch.nn import init


import time


import numpy as np


from torch.nn import functional as F


import logging


class C3D(nn.Module):

    def __init__(self, sample_size, sample_duration, num_classes=600):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.BatchNorm3d(256), nn.ReLU(), nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.BatchNorm3d(256), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))
        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(nn.Linear(512 * last_duration * last_size * last_size, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))


class MobileNet(nn.Module):

    def __init__(self, num_classes=600, sample_size=224, width_mult=1.0):
        super(MobileNet, self).__init__()
        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [[64, 1, (2, 2, 2)], [128, 2, (2, 2, 2)], [256, 2, (2, 2, 2)], [512, 6, (2, 2, 2)], [1024, 2, (1, 1, 1)]]
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv3d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup), nn.ReLU(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, sample_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, (1, 1, 1)], [6, 24, 2, (2, 2, 2)], [6, 32, 3, (2, 2, 2)], [6, 64, 4, (2, 2, 2)], [6, 96, 3, (1, 1, 1)], [6, 160, 3, (2, 2, 2)], [6, 320, 1, (1, 1, 1)]]
        assert sample_size % 16 == 0.0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNeXt(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def channel_shuffle(x, groups):
    """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class Bottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes // 4
        if self.stride == 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.stride == 2:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)
        return out


class ShuffleNet(nn.Module):

    def __init__(self, groups, width_mult=1, num_classes=400):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.groups = groups
        num_blocks = [4, 8, 4]
        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError("""{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))
        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1 = conv_bn(3, self.in_planes, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(out_planes[1], num_blocks[0], self.groups)
        self.layer2 = self._make_layer(out_planes[2], num_blocks[1], self.groups)
        self.layer3 = self._make_layer(out_planes[3], num_blocks[2], self.groups)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(out_planes[3], self.num_classes))

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2
        if self.stride == 1:
            self.banch2 = nn.Sequential(nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm3d(oup_inc), nn.ReLU(inplace=True), nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm3d(oup_inc), nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm3d(oup_inc), nn.ReLU(inplace=True))
        else:
            self.banch1 = nn.Sequential(nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm3d(inp), nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm3d(oup_inc), nn.ReLU(inplace=True))
            self.banch2 = nn.Sequential(nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm3d(oup_inc), nn.ReLU(inplace=True), nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm3d(oup_inc), nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm3d(oup_inc), nn.ReLU(inplace=True))

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :x.shape[1] // 2, :, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):

    def __init__(self, num_classes=600, sample_size=112, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0
        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24, 32, 64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.conv_last = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.stage_out_channels[-1], num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)
        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)
        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
            out += x
        out = self.relu(out)
        return out


class SqueezeNet(nn.Module):

    def __init__(self, sample_size, sample_duration, version=1.1, num_classes=600):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv3d(3, 96, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3)), nn.BatchNorm3d(96), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64, use_bypass=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128, use_bypass=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192, use_bypass=True), Fire(384, 64, 256, 256), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(512, 64, 256, 256, use_bypass=True))
        if version == 1.1:
            self.features = nn.Sequential(nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64, use_bypass=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128, use_bypass=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192, use_bypass=True), nn.MaxPool3d(kernel_size=3, stride=2, padding=1), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256, use_bypass=True))
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool3d((last_duration, last_size, last_size), stride=1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Block,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (ShuffleNet,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     False),
    (ShuffleNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     False),
    (SqueezeNet,
     lambda: ([], {'sample_size': 4, 'sample_duration': 4}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     True),
]

class Test_okankop_Efficient_3DCNNs(_paritybench_base):
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

