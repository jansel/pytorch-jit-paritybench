import sys
_module = sys.modules[__name__]
del sys
pytorch2keras = _module
converter = _module
setup = _module
tests = _module
layers = _module
hard_tanh = _module
lrelu = _module
relu = _module
selu = _module
sigmoid = _module
softmax = _module
tanh = _module
constant = _module
conv1d = _module
conv2d = _module
convtranspose2d = _module
add = _module
div = _module
mul = _module
sub = _module
embedding = _module
linear = _module
multiple_inputs = _module
bn2d = _module
do = _module
in2d = _module
avgpool2d = _module
global_avgpool2d = _module
global_maxpool2d = _module
maxpool2d = _module
upsample_nearest = _module
upsampling_bilinear = _module
upsampling_nearest = _module
models = _module
alexnet = _module
drn = _module
menet = _module
mobilinet = _module
preresnet18 = _module
resnet18 = _module
resnet18_channels_last = _module
resnet34 = _module
resnet50 = _module
senet = _module
squeezenet = _module
squeezenext = _module
vgg11 = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.autograd import Variable


import random


import torch.nn.functional as F


import math


import torch.nn.init as init


from torch import nn


from torchvision.models import ResNet


import torchvision


class LayerTest(nn.Module):

    def __init__(self, min_val, max_val):
        super(LayerTest, self).__init__()
        self.htanh = nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x):
        x = self.htanh(x)
        return x


class FTest(nn.Module):

    def __init__(self, min_val, max_val):
        super(FTest, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        from torch.nn import functional as F
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


class LayerTest(nn.Module):

    def __init__(self, negative_slope):
        super(LayerTest, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        x = self.relu(x)
        return x


class FTest(nn.Module):

    def __init__(self, negative_slope):
        super(FTest, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        from torch.nn import functional as F
        return F.leaky_relu(x, self.negative_slope)


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.relu(x)


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.selu(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.selu(x)


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.sigmoid(x)


class LayerTest(nn.Module):

    def __init__(self, dim):
        super(LayerTest, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x = self.softmax(x)
        return x


class FTest(nn.Module):

    def __init__(self, dim):
        super(FTest, self).__init__()
        self.dim = dim

    def forward(self, x):
        from torch.nn import functional as F
        return F.softmax(x, dim=self.dim)


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.tanh(x)


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        return x + torch.FloatTensor([1.0])


class LayerTest(nn.Module):

    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False, dilation=1):
        super(LayerTest, self).__init__()
        self.conv = nn.Conv1d(inp, out, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False, dilation=1, groups=1):
        super(LayerTest, self).__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False):
        super(LayerTest, self).__init__()
        self.conv = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x + y
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x / y
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x * y
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x - y
        return x


class LayerTest(nn.Module):

    def __init__(self, input_size, embedd_size):
        super(LayerTest, self).__init__()
        self.embedd = nn.Embedding(input_size, embedd_size)

    def forward(self, x):
        x = self.embedd(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, inp, out, bias=False):
        super(LayerTest, self).__init__()
        self.fc = nn.Linear(inp, out, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y, z):
        from torch.nn import functional as F
        return F.relu(x) + F.relu(y) + F.relu(z)


class LayerTest(nn.Module):

    def __init__(self, out, eps, momentum):
        super(LayerTest, self).__init__()
        self.bn = nn.BatchNorm2d(out, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.bn(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, p):
        super(LayerTest, self).__init__()
        self.do = nn.Dropout2d(p=p)

    def forward(self, x):
        x = x + 0
        x = self.do(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, out, eps, momentum):
        super(LayerTest, self).__init__()
        self.in2d = nn.InstanceNorm2d(out, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.in2d(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, kernel_size=3, padding=1, stride=1):
        super(LayerTest, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x


class LayerTest(nn.Module):

    def __init__(self):
        super(LayerTest, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, kernel_size=3, padding=1, stride=1):
        super(LayerTest, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x


class TestUpsampleNearest2d(nn.Module):
    """Module for UpsampleNearest2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestUpsampleNearest2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.upsample(x, scale_factor=2)
        x = self.up(x)
        return x


class LayerTest(nn.Module):

    def __init__(self, scale_factor=2):
        super(LayerTest, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        x = self.up(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.upsample_bilinear(x, scale_factor=2)


class LayerTest(nn.Module):

    def __init__(self, scale_factor=2):
        super(LayerTest, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, x):
        x = self.up(x)
        return x


class FTest(nn.Module):

    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.upsample_nearest(x, scale_factor=2)


BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
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


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), BatchNorm(channels[0]), nn.ReLU(inplace=True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False)
        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7], dilation=1)
        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm(planes * block.expansion))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation), residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual, dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(self.inplanes, channels, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), BatchNorm(channels), nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)
        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)
        x = self.layer3(x)
        y.append(x)
        x = self.layer4(x)
        y.append(x)
        x = self.layer5(x)
        y.append(x)
        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)
        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
        if self.out_middle:
            return x, y
        else:
            return x


class DRN_A(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AvgPool2d(28, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))
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
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]
    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


class ShuffleInitBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ShuffleInitBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


def conv1x1(in_channels, out_channels, stride):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def depthwise_conv3x3(channels, stride):
    return nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)


def group_conv1x1(in_channels, out_channels, groups):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=groups, bias=False)


class MEModule(nn.Module):

    def __init__(self, in_channels, out_channels, side_channels, groups, downsample, ignore_group):
        super(MEModule, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4
        if downsample:
            out_channels -= in_channels
        self.compress_conv1 = group_conv1x1(in_channels=in_channels, out_channels=mid_channels, groups=1 if ignore_group else groups)
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(channels=mid_channels, groups=1 if ignore_group else groups)
        self.dw_conv2 = depthwise_conv3x3(channels=mid_channels, stride=2 if self.downsample else 1)
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = group_conv1x1(in_channels=mid_channels, out_channels=out_channels, groups=groups)
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = nn.ReLU(inplace=True)
        self.s_merge_conv = conv1x1(in_channels=mid_channels, out_channels=side_channels)
        self.s_merge_bn = nn.BatchNorm2d(num_features=side_channels)
        self.s_conv = conv3x3(in_channels=side_channels, out_channels=side_channels, stride=2 if self.downsample else 1)
        self.s_conv_bn = nn.BatchNorm2d(num_features=side_channels)
        self.s_evolve_conv = conv1x1(in_channels=side_channels, out_channels=mid_channels)
        self.s_evolve_bn = nn.BatchNorm2d(num_features=mid_channels)

    def forward(self, x):
        identity = x
        x = self.activ(self.compress_bn1(self.compress_conv1(x)))
        x = self.c_shuffle(x)
        y = self.s_merge_conv(x)
        y = self.s_merge_bn(y)
        y = self.activ(y)
        x = self.dw_bn2(self.dw_conv2(x))
        y = self.s_conv(y)
        y = self.s_conv_bn(y)
        y = self.activ(y)
        y = self.s_evolve_conv(y)
        y = self.s_evolve_bn(y)
        y = F.sigmoid(y)
        x = x * y
        x = self.expand_bn3(self.expand_conv3(x))
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


class MENet(nn.Module):

    def __init__(self, block_channels, side_channels, groups, num_classes=1000):
        super(MENet, self).__init__()
        input_channels = 3
        block_layers = [4, 8, 4]
        self.features = nn.Sequential()
        self.features.add_module('init_block', ShuffleInitBlock(in_channels=input_channels, out_channels=block_channels[0]))
        for i in range(len(block_channels) - 1):
            stage = nn.Sequential()
            in_channels_i = block_channels[i]
            out_channels_i = block_channels[i + 1]
            for j in range(block_layers[i]):
                stage.add_module('unit_{}'.format(j + 1), MEModule(in_channels=in_channels_i if j == 0 else out_channels_i, out_channels=out_channels_i, side_channels=side_channels, groups=groups, downsample=j == 0, ignore_group=i == 0 and j == 0))
            self.features.add_module('stage_{}'.format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7))
        self.output = nn.Linear(in_features=block_channels[-1], out_features=num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 32))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(self.last_channel, n_class))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PreResConv(nn.Module):
    """
    PreResNet specific convolution block, with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PreResConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x_pre_activ = x
        x = self.conv(x)
        return x, x_pre_activ


def preres_conv3x3(in_channels, out_channels, stride):
    """
    3x3 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return PreResConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)


class PreResBlock(nn.Module):
    """
    Simple PreResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(PreResBlock, self).__init__()
        self.conv1 = preres_conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = preres_conv3x3(in_channels=out_channels, out_channels=out_channels, stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        return x, x_pre_activ


def preres_conv1x1(in_channels, out_channels, stride):
    """
    1x1 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return PreResConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)


class PreResBottleneck(nn.Module):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """

    def __init__(self, in_channels, out_channels, stride, conv1_stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = preres_conv1x1(in_channels=in_channels, out_channels=mid_channels, stride=stride if conv1_stride else 1)
        self.conv2 = preres_conv3x3(in_channels=mid_channels, out_channels=mid_channels, stride=1 if conv1_stride else stride)
        self.conv3 = preres_conv1x1(in_channels=mid_channels, out_channels=out_channels, stride=1)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x, _ = self.conv2(x)
        x, _ = self.conv3(x)
        return x, x_pre_activ


class PreResUnit(nn.Module):
    """
    PreResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """

    def __init__(self, in_channels, out_channels, stride, bottleneck, conv1_stride):
        super(PreResUnit, self).__init__()
        self.resize_identity = in_channels != out_channels or stride != 1
        if bottleneck:
            self.body = PreResBottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, conv1_stride=conv1_stride)
        else:
            self.body = PreResBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(nn.Module):
    """
    PreResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(PreResInitBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class PreResActivation(nn.Module):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """

    def __init__(self, in_channels):
        super(PreResActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class PreResNet(nn.Module):
    """
    PreResNet model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self, channels, init_block_channels, bottleneck, conv1_stride, in_channels=3, num_classes=1000):
        super(PreResNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('init_block', PreResInitBlock(in_channels=in_channels, out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if i == 0 or j != 0 else 2
                stage.add_module('unit{}'.format(j + 1), PreResUnit(in_channels=in_channels, out_channels=out_channels, stride=stride, bottleneck=bottleneck, conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.features.add_module('post_activ', PreResActivation(in_channels=in_channels))
        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7, stride=1))
        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(int(x.size(0)), -1)
        x = self.output(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view([int(b), -1])
        y = self.fc(y)
        y = y.view([int(b), int(c), 1, 1])
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CifarSEBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class CifarSEResNet(nn.Module):

    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view([int(x.size(0)), -1])
        x = self.fc(x)
        return x


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

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13, stride=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view([int(x.size(0)), self.num_classes])


class SqnxtConv(nn.Module):
    """
    SqueezeNext specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default (0, 0)
        Padding value for convolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
        super(SqnxtConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class SqnxtUnit(nn.Module):
    """
    SqueezeNext unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(SqnxtUnit, self).__init__()
        if stride == 2:
            reduction_den = 1
            self.resize_identity = True
        elif in_channels > out_channels:
            reduction_den = 4
            self.resize_identity = True
        else:
            reduction_den = 2
            self.resize_identity = False
        self.conv1 = SqnxtConv(in_channels=in_channels, out_channels=in_channels // reduction_den, kernel_size=1, stride=stride)
        self.conv2 = SqnxtConv(in_channels=in_channels // reduction_den, out_channels=in_channels // (2 * reduction_den), kernel_size=1, stride=1)
        self.conv3 = SqnxtConv(in_channels=in_channels // (2 * reduction_den), out_channels=in_channels // reduction_den, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = SqnxtConv(in_channels=in_channels // reduction_den, out_channels=in_channels // reduction_den, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv5 = SqnxtConv(in_channels=in_channels // reduction_den, out_channels=out_channels, kernel_size=1, stride=1)
        if self.resize_identity:
            self.identity_conv = SqnxtConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        identity = self.activ(identity)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x


class SqnxtInitBlock(nn.Module):
    """
    SqueezeNext specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(SqnxtInitBlock, self).__init__()
        self.conv = SqnxtConv(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SqueezeNext(nn.Module):
    """
    SqueezeNext model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self, channels, init_block_channels, final_block_channels, in_channels=3, num_classes=1000):
        super(SqueezeNext, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('init_block', SqnxtInitBlock(in_channels=in_channels, out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if j == 0 and i != 0 else 1
                stage.add_module('unit{}'.format(j + 1), SqnxtUnit(in_channels=in_channels, out_channels=out_channels, stride=stride))
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.features.add_module('final_block', SqnxtConv(in_channels=in_channels, out_channels=final_block_channels, kernel_size=1, stride=1))
        in_channels = final_block_channels
        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7, stride=1))
        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChannelShuffle,
     lambda: ([], {'channels': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CifarSEBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 16}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FTest,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerTest,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (PreResActivation,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreResBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'conv1_stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreResConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreResInitBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreResUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'bottleneck': 4, 'conv1_stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (ShuffleInitBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqnxtConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqnxtInitBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (SqnxtUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TestUpsampleNearest2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10, 64, 64])], {}),
     False),
]

class Test_nerox8664_pytorch2keras(_paritybench_base):
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

