import sys
_module = sys.modules[__name__]
del sys
main = _module
EfficientFace = _module
modulator = _module
resnet = _module

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


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import matplotlib.pyplot as plt


import torchvision.datasets as datasets


import torchvision.transforms as transforms


import numpy as np


import torch.nn.functional as F


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


class LocalFeatureExtractor(nn.Module):

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index
        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)
        self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)
        self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)
        self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)

    def forward(self, x):
        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]
        out_1 = self.conv1_1(patch_11)
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)
        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)
        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Channel(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(Channel, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class Spatial(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(Spatial, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class Modulator(nn.Module):

    def __init__(self, gate_channel):
        super(Modulator, self).__init__()
        self.channel_att = Channel(gate_channel)
        self.spatial_att = Spatial(gate_channel)

    def forward(self, in_tensor):
        att = torch.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


class EfficientFace(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7):
        super(EfficientFace, self).__init__()
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LocalFeatureExtractor,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'index': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_zengqunzhao_EfficientFace(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

