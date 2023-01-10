import sys
_module = sys.modules[__name__]
del sys
SE_block = _module
models = _module
SE_block = _module
blocks = _module
conv_bn = _module
dense_block = _module
dpn_block = _module
inception_blocks = _module
residual_blocks = _module
resnext_block = _module
shuffle_block = _module
csp_darknet = _module
csp_resnext = _module
darknet = _module
densenet = _module
dpn = _module
googlenet = _module
inception_blocks = _module
inception_v3 = _module
inception_v4_resnet = _module
mnasnet = _module
resnet = _module
resnext = _module
shufflenet_v2 = _module
vgg = _module
onnx_infer = _module
predict = _module
test_net = _module
to_onnx = _module
train = _module
utils = _module
arg_utils = _module
data_utils = _module
earlystopping = _module
progress_utils = _module
vino_infer = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


import torchvision


import torchvision.transforms as transforms


import torch.backends.cudnn as cudnn


import time


import torch.optim as optim


import numpy as np


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object, dilation=1, groups=1, bias=False) ->object:
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.relu(self.seq(x))


class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object, dilation=1, groups=1, bias=False) ->object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.leaky_relu(self.seq(x))


class Mish(nn.Module):

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BN_Conv_Mish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return Mish()(out)


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0), BN_Conv2d(4 * self.k, self.k, 3, 1, 1)))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class CSP_DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, k, part_ratio=0.5):
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.dense = DenseBlock(self.part2_chnls, num_layers, k)

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        part2 = self.dense(part2)
        out = torch.cat((part1, part2), 1)
        return out


class DPN_Block(nn.Module):
    """
    Dual Path block
    """

    def __init__(self, in_chnls, add_chnl, cat_chnl, cardinality, d, stride):
        super(DPN_Block, self).__init__()
        self.add = add_chnl
        self.cat = cat_chnl
        self.chnl = cardinality * d
        self.conv1 = BN_Conv2d(in_chnls, self.chnl, 1, 1, 0)
        self.conv2 = BN_Conv2d(self.chnl, self.chnl, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.chnl, add_chnl + cat_chnl, 1, 1, 0)
        self.bn = nn.BatchNorm2d(add_chnl + cat_chnl)
        self.shortcut = nn.Sequential()
        if add_chnl != in_chnls:
            self.shortcut = nn.Sequential(nn.Conv2d(in_chnls, add_chnl, 1, stride, 0), nn.BatchNorm2d(add_chnl))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        add = out[:, :self.add, :, :] + self.shortcut(x)
        out = torch.cat((add, out[:, self.add:, :, :]), dim=1)
        return F.relu(out)


class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(BN_Conv2d(3, 32, 3, 2, 0, bias=False), BN_Conv2d(32, 32, 3, 1, 0, bias=False), BN_Conv2d(32, 64, 3, 1, 1, bias=False))
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(BN_Conv2d(160, 64, 1, 1, 0, bias=False), BN_Conv2d(64, 96, 3, 1, 0, bias=False))
        self.step3_2 = nn.Sequential(BN_Conv2d(160, 64, 1, 1, 0, bias=False), BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(64, 96, 3, 1, 0, bias=False))
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        None
        None
        out = torch.cat((tmp1, tmp2), 1)
        return out


class Stem_Res1(nn.Module):
    """
    stem block for Inception-ResNet-v1
    """

    def __init__(self):
        super(Stem_Res1, self).__init__()
        self.stem = nn.Sequential(BN_Conv2d(3, 32, 3, 2, 0, bias=False), BN_Conv2d(32, 32, 3, 1, 0, bias=False), BN_Conv2d(32, 64, 3, 1, 1, bias=False), nn.MaxPool2d(3, 2, 0), BN_Conv2d(64, 80, 1, 1, 0, bias=False), BN_Conv2d(80, 192, 3, 1, 0, bias=False), BN_Conv2d(192, 256, 3, 2, 0, bias=False))

    def forward(self, x):
        return self.stem(x)


class Inception_A(nn.Module):
    """
    Inception-A block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(3, 1, 1), BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False))
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False))
        self.branch4 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False), BN_Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1, b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(3, 1, 1), BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False))
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False))
        self.branch4 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1, b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(3, 1, 1), BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False))
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False), BN_Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False))
        self.branch4_1x3 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)


class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, k, 1, 1, 0, bias=False), BN_Conv2d(k, l, 3, 1, 1, bias=False), BN_Conv2d(l, m, 3, 2, 0, bias=False))

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False))

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Reduction_B_Res(nn.Module):
    """
    Reduction-B block for Inception-ResNet-v1     and Inception-ResNet-v1  net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n3, b4_n1, b4_n3_1, b4_n3_2):
        super(Reduction_B_Res, self).__init__()
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3, 3, 2, 0, bias=False))
        self.branch4 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n3_1, 3, 1, 1, bias=False), BN_Conv2d(b4_n3_1, b4_n3_2, 3, 2, 0, bias=False))

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_A_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False), BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n3 + b3_n3_2, n1_linear, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Inception_B_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x7, b2_n7x1, n1_linear):
        super(Inception_B_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n1x7, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b2_n1x7, b2_n7x1, (7, 1), (1, 1), (3, 0), bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n7x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Inception_C_res(nn.Module):
    """
    Inception-C block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x3, b2_n3x1, n1_linear):
        super(Inception_C_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n1x3, (1, 3), (1, 1), (0, 1), bias=False), BN_Conv2d(b2_n1x3, b2_n3x1, (3, 1), (1, 1), (1, 0), bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n3x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = 'basic'

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)


class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = 'bottleneck'

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False), nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return F.relu(out)


class Dark_block(nn.Module):
    """block for darknet"""

    def __init__(self, channels, is_se=False, inner_channels=None):
        super(Dark_block, self).__init__()
        self.is_se = is_se
        if inner_channels is None:
            inner_channels = channels // 2
        self.conv1 = BN_Conv2d_Leaky(channels, inner_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(inner_channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        if self.is_se:
            self.se = SE(channels, 16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += x
        return F.leaky_relu(out)


class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride, is_se=False):
        super(ResNeXt_Block, self).__init__()
        self.is_se = is_se
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls * 2)
        if self.is_se:
            self.se = SE(self.group_chnls * 2, 16)
        self.short_cut = nn.Sequential(nn.Conv2d(in_chnls, self.group_chnls * 2, 1, stride, 0, bias=False), nn.BatchNorm2d(self.group_chnls * 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.short_cut(x)
        return F.relu(out)


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""
    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1, groups=self.ro_chnls, activation=False)
        act = False if self.is_res else True
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        x_l = x[:, :self.l_chnls, :, :]
        x_r = x[:, self.l_chnls:, :, :]
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=False)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=False)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)
        out = torch.cat((out_l, out_r), 1)
        out = shuffle_chnls(out, self.groups)
        return shuffle_chnls(out, self.groups)


class ResidualBlock(nn.Module):
    """
    Residual block for CSP-ResNeXt
    """

    def __init__(self, in_channels, cardinality, group_width, stride=1):
        super(ResidualBlock, self).__init__()
        self.out_channels = cardinality * group_width
        self.conv1 = BN_Conv2d_Leaky(in_channels, self.out_channels, 1, 1, 0)
        self.conv2 = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(self.out_channels)
        layers = []
        if in_channels != self.out_channels:
            layers.append(nn.Conv2d(in_channels, self.out_channels, 1, 1, 0))
            layers.append(nn.BatchNorm2d(self.out_channels))
        if stride != 1:
            layers.append(nn.AvgPool2d(stride))
        self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.bn(out)
        out += self.shortcut(x)
        return F.leaky_relu(out)


class CSPFirst(nn.Module):
    """
    First CSP Stage
    """

    def __init__(self, in_chnnls, out_chnls):
        super(CSPFirst, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.block = ResidualBlock(out_chnls, out_chnls // 2)
        self.trans_cat = BN_Conv_Mish(2 * out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.block(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSPStem(nn.Module):
    """
    CSP structures including downsampling
    """

    def __init__(self, in_chnls, out_chnls, num_block):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_chnls // 2) for _ in range(num_block)])
        self.trans_cat = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSP_DarkNet(nn.Module):
    """
    CSP-DarkNet
    """

    def __init__(self, num_blocks: object, num_classes=1000) ->object:
        super(CSP_DarkNet, self).__init__()
        chnls = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3, 32, 3, 1, 1)
        self.neck = CSPFirst(32, chnls[0])
        self.body = nn.Sequential(*[CSPStem(chnls[i], chnls[i + 1], num_blocks[i]) for i in range(4)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(chnls[4], num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Stem(nn.Module):

    def __init__(self, in_channels, num_blocks, cardinality, group_with, stride=2):
        super(Stem, self).__init__()
        self.c0 = in_channels // 2
        self.c1 = in_channels - in_channels // 2
        self.hidden_channels = cardinality * group_with
        self.out_channels = self.hidden_channels * 2
        self.transition = BN_Conv2d_Leaky(self.hidden_channels, in_channels, 1, 1, 0)
        self.trans_part0 = nn.Sequential(BN_Conv2d_Leaky(self.c0, self.hidden_channels, 1, 1, 0), nn.AvgPool2d(stride))
        self.block = self.__make_block(num_blocks, self.c1, cardinality, group_with, stride)
        self.trans_part1 = BN_Conv2d_Leaky(self.hidden_channels, self.hidden_channels, 1, 1, 0)
        self.trans = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 1, 1, 0)

    def __make_block(self, num_blocks, in_channels, cardinality, group_with, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        channels = [in_channels] + [self.hidden_channels] * (num_blocks - 1)
        return nn.Sequential(*[ResidualBlock(c, cardinality, group_with, s) for c, s in zip(channels, strides)])

    def forward(self, x):
        x = self.transition(x)
        x0 = x[:, :self.c0, :, :]
        x1 = x[:, self.c0:, :, :]
        out0 = self.trans_part0(x0)
        out1 = self.trans_part1(self.block(x1))
        out = torch.cat((out0, out1), 1)
        return self.trans(out)


class CSP_ResNeXt(nn.Module):

    def __init__(self, num_blocks, cadinality, group_width, num_classes):
        super(CSP_ResNeXt, self).__init__()
        self.conv0 = BN_Conv2d_Leaky(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv1 = BN_Conv2d_Leaky(64, 128, 1, 1, 0)
        self.stem0 = Stem(cadinality * group_width * 2, num_blocks[0], cadinality, group_width, stride=1)
        self.stem1 = Stem(cadinality * group_width * 4, num_blocks[1], cadinality, group_width * 2)
        self.stem2 = Stem(cadinality * group_width * 8, num_blocks[2], cadinality, group_width * 4)
        self.stem3 = Stem(cadinality * group_width * 16, num_blocks[3], cadinality, group_width * 8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cadinality * group_width * 16, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.pool1(out)
        out = self.conv1(out)
        out = self.stem0(out)
        out = self.stem1(out)
        out = self.stem2(out)
        out = self.stem3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DarkNet(nn.Module):

    def __init__(self, layers: object, num_classes, is_se=False) ->object:
        super(DarkNet, self).__init__()
        self.is_se = is_se
        filters = [64, 128, 256, 512, 1024]
        self.conv1 = BN_Conv2d(3, 32, 3, 1, 1)
        self.redu1 = BN_Conv2d(32, 64, 3, 2, 1)
        self.conv2 = self.__make_layers(filters[0], layers[0])
        self.redu2 = BN_Conv2d(filters[0], filters[1], 3, 2, 1)
        self.conv3 = self.__make_layers(filters[1], layers[1])
        self.redu3 = BN_Conv2d(filters[1], filters[2], 3, 2, 1)
        self.conv4 = self.__make_layers(filters[2], layers[2])
        self.redu4 = BN_Conv2d(filters[2], filters[3], 3, 2, 1)
        self.conv5 = self.__make_layers(filters[3], layers[3])
        self.redu5 = BN_Conv2d(filters[3], filters[4], 3, 2, 1)
        self.conv6 = self.__make_layers(filters[4], layers[4])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[4], num_classes)

    def __make_layers(self, num_filter, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(Dark_block(num_filter, self.is_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.redu1(out)
        out = self.conv2(out)
        out = self.redu2(out)
        out = self.conv3(out)
        out = self.redu3(out)
        out = self.conv4(out)
        out = self.redu4(out)
        out = self.conv5(out)
        out = self.redu5(out)
        out = self.conv6(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DenseNet(nn.Module):

    def __init__(self, layers: object, k, theta, num_classes, part_ratio=0) ->object:
        super(DenseNet, self).__init__()
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio == 0 else CSP_DenseBlock
        self.conv = BN_Conv2d(3, 2 * k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2 * k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transition(self, in_chls):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(BN_Conv2d(in_chls, out_chls, 1, 1, 0), nn.AvgPool2d(2)), out_chls

    def __make_blocks(self, k0):
        """
        make block-transition structures
        :param k0:
        :return:
        """
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(self.Block(k0, self.layers[i], self.k))
            patches = k0 + self.layers[i] * self.k
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.blocks(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DPN(nn.Module):

    def __init__(self, blocks: object, add_chnls: object, cat_chnls: object, conv1_chnl, cardinality, d, num_classes) ->object:
        super(DPN, self).__init__()
        self.cdty = cardinality
        self.chnl = conv1_chnl
        self.conv1 = BN_Conv2d(3, self.chnl, 7, 2, 3)
        d1 = d
        self.conv2 = self.__make_layers(blocks[0], add_chnls[0], cat_chnls[0], d1, 1)
        d2 = 2 * d1
        self.conv3 = self.__make_layers(blocks[1], add_chnls[1], cat_chnls[1], d2, 2)
        d3 = 2 * d2
        self.conv4 = self.__make_layers(blocks[2], add_chnls[2], cat_chnls[2], d3, 2)
        d4 = 2 * d3
        self.conv5 = self.__make_layers(blocks[3], add_chnls[3], cat_chnls[3], d4, 2)
        self.fc = nn.Linear(self.chnl, num_classes)

    def __make_layers(self, block, add_chnl, cat_chnl, d, stride):
        layers = []
        strides = [stride] + [1] * (block - 1)
        for i, s in enumerate(strides):
            layers.append(DPN_Block(self.chnl, add_chnl, cat_chnl, self.cdty, d, s))
            self.chnl = add_chnl + cat_chnl
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Inception_builder(nn.Module):
    """
    types of Inception block
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Inception_builder, self).__init__()
        self.block_type = block_type
        self.branch1_type1 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b1_reduce, b1, 5, stride=1, padding=2, bias=False))
        self.branch1_type2 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b1_reduce, b1, 3, stride=1, padding=1, bias=False), BN_Conv2d(b1, b1, 3, stride=1, padding=1, bias=False))
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b2_reduce, b2, 3, stride=1, padding=1, bias=False))
        self.branch3 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), BN_Conv2d(in_channels, b3, 1, stride=1, padding=0, bias=False))
        self.branch4 = BN_Conv2d(in_channels, b4, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.block_type == 'type1':
            out1 = self.branch1_type1(x)
            out2 = self.branch2(x)
        elif self.block_type == 'type2':
            out1 = self.branch1_type2(x)
            out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)
        return out


class GoogleNet(nn.Module):
    """
    Inception-v1, Inception-v2
    """

    def __init__(self, str_version, num_classes):
        super(GoogleNet, self).__init__()
        self.block_type = 'type1' if str_version == 'v1' else 'type2'
        self.version = str_version
        self.conv1 = BN_Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.conv2 = BN_Conv2d(64, 192, 3, stride=1, padding=1, bias=False)
        self.inception3_a = Inception_builder(self.block_type, 192, 16, 32, 96, 128, 32, 64)
        self.inception3_b = Inception_builder(self.block_type, 256, 32, 96, 128, 192, 64, 128)
        self.inception4_a = Inception_builder(self.block_type, 480, 16, 48, 96, 208, 64, 192)
        self.inception4_b = Inception_builder(self.block_type, 512, 24, 64, 112, 224, 64, 160)
        self.inception4_c = Inception_builder(self.block_type, 512, 24, 64, 128, 256, 64, 128)
        self.inception4_d = Inception_builder(self.block_type, 512, 32, 64, 144, 288, 64, 112)
        self.inception4_e = Inception_builder(self.block_type, 528, 32, 128, 160, 320, 128, 256)
        self.inception5_a = Inception_builder(self.block_type, 832, 32, 128, 160, 320, 128, 256)
        self.inception5_b = Inception_builder(self.block_type, 832, 48, 128, 192, 384, 128, 384)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception3_a(out)
        out = self.inception3_b(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception4_a(out)
        out = self.inception4_b(out)
        out = self.inception4_c(out)
        out = self.inception4_d(out)
        out = self.inception4_e(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception5_a(out)
        out = self.inception5_b(out)
        out = F.avg_pool2d(out, 7)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Block_bank(nn.Module):
    """
    inception structures
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Block_bank, self).__init__()
        self.block_type = block_type
        """
        branch 1
        """
        self.branch1_type1 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False), BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False), BN_Conv2d(b1, b1, 3, 1, 1, bias=False))
        self.branch1_type2 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False), BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False), BN_Conv2d(b1, b1, 3, 2, 0, bias=False))
        self.branch1_type3 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False), BN_Conv2d(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b1_reduce, b1_reduce, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b1_reduce, b1, (1, 7), (1, 1), (0, 3), bias=False))
        self.branch1_type4 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False), BN_Conv2d(b1_reduce, b1, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b1, b1, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b1, b1, 3, 2, 0, bias=False))
        self.branch1_type5_head = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, 1, 0, bias=False), BN_Conv2d(b1_reduce, b1, 3, 1, 1, bias=False))
        self.branch1_type5_body1 = BN_Conv2d(b1, b1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch1_type5_body2 = BN_Conv2d(b1, b1, (3, 1), (1, 1), (1, 0), bias=False)
        """
        branch 2
        """
        self.branch2_type1 = nn.Sequential(BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False), BN_Conv2d(b2_reduce, b2, 5, 1, 2, bias=False))
        self.branch2_type2 = BN_Conv2d(in_channels, b2, 3, 2, 0, bias=False)
        self.branch2_type3 = nn.Sequential(BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False), BN_Conv2d(b2_reduce, b2_reduce, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b2_reduce, b2, (7, 1), (1, 1), (3, 0), bias=False))
        self.branch2_type4 = nn.Sequential(BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False), BN_Conv2d(b2_reduce, b2, 3, 2, 0, bias=False))
        self.branch2_type5_head = BN_Conv2d(in_channels, b2_reduce, 1, 1, 0, bias=False)
        self.branch2_type5_body1 = BN_Conv2d(b2_reduce, b2, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch2_type5_body2 = BN_Conv2d(b2_reduce, b2, (3, 1), (1, 1), (1, 0), bias=False)
        """
        branch 3
        """
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, 1, 1), BN_Conv2d(in_channels, b3, 1, 1, 0, bias=False))
        """
        branch 4
        """
        self.branch4 = BN_Conv2d(in_channels, b4, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.block_type == 'type1':
            out1 = self.branch1_type1(x)
            out2 = self.branch2_type1(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1, out2, out3, out4), 1)
        elif self.block_type == 'type2':
            out1 = self.branch1_type2(x)
            out2 = self.branch2_type2(x)
            out3 = F.max_pool2d(x, 3, 2, 0)
            out = torch.cat((out1, out2, out3), 1)
        elif self.block_type == 'type3':
            out1 = self.branch1_type3(x)
            out2 = self.branch2_type3(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1, out2, out3, out4), 1)
        elif self.block_type == 'type4':
            out1 = self.branch1_type4(x)
            out2 = self.branch2_type4(x)
            out3 = F.max_pool2d(x, 3, 2, 0)
            out = torch.cat((out1, out2, out3), 1)
        else:
            tmp = self.branch1_type5_head(x)
            out1_1 = self.branch1_type5_body1(tmp)
            out1_2 = self.branch1_type5_body2(tmp)
            tmp = self.branch2_type5_head(x)
            out2_1 = self.branch2_type5_body1(tmp)
            out2_2 = self.branch2_type5_body2(tmp)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            out = torch.cat((out1_1, out1_2, out2_1, out2_2, out3, out4), 1)
        return out


class Inception_v3(nn.Module):

    def __init__(self, num_classes):
        super(Inception_v3, self).__init__()
        self.conv = BN_Conv2d(3, 32, 3, 2, 0, bias=False)
        self.conv1 = BN_Conv2d(32, 32, 3, 1, 0, bias=False)
        self.conv2 = BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        self.conv3 = BN_Conv2d(64, 80, 1, 1, 0, bias=False)
        self.conv4 = BN_Conv2d(80, 192, 3, 1, 0, bias=False)
        self.inception1_1 = Block_bank('type1', 192, 64, 96, 48, 64, 32, 64)
        self.inception1_2 = Block_bank('type1', 256, 64, 96, 48, 64, 64, 64)
        self.inception1_3 = Block_bank('type1', 288, 64, 96, 48, 64, 64, 64)
        self.inception2 = Block_bank('type2', 288, 64, 96, 288, 384, 288, 288)
        self.inception3_1 = Block_bank('type3', 768, 128, 192, 128, 192, 192, 192)
        self.inception3_2 = Block_bank('type3', 768, 160, 192, 160, 192, 192, 192)
        self.inception3_3 = Block_bank('type3', 768, 160, 192, 160, 192, 192, 192)
        self.inception3_4 = Block_bank('type3', 768, 192, 192, 192, 192, 192, 192)
        self.inception4 = Block_bank('type4', 768, 192, 192, 192, 320, 288, 288)
        self.inception5_1 = Block_bank('type5', 1280, 448, 384, 384, 384, 192, 320)
        self.inception5_2 = Block_bank('type5', 2048, 448, 384, 384, 384, 192, 320)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 0)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.max_pool2d(out, 3, 2, 0)
        out = self.inception1_1(out)
        out = self.inception1_2(out)
        out = self.inception1_3(out)
        out = self.inception2(out)
        out = self.inception3_1(out)
        out = self.inception3_2(out)
        out = self.inception3_3(out)
        out = self.inception3_4(out)
        out = self.inception4(out)
        out = self.inception5_1(out)
        out = self.inception5_2(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Inception(nn.Module):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, version, num_classes, is_se=False):
        super(Inception, self).__init__()
        self.version = version
        self.stem = Stem_Res1() if self.version == 'res1' else Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        if self.version == 'v4':
            self.fc = nn.Linear(1536, num_classes)
        elif self.version == 'res1':
            self.fc = nn.Linear(1792, num_classes)
        else:
            self.fc = nn.Linear(2144, num_classes)

    def __make_inception_A(self):
        layers = []
        if self.version == 'v4':
            for _ in range(4):
                layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == 'res1':
            for _ in range(5):
                layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(5):
                layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == 'v4':
            return Reduction_A(384, 192, 224, 256, 384)
        elif self.version == 'res1':
            return Reduction_A(256, 192, 192, 256, 384)
        else:
            return Reduction_A(384, 256, 256, 384, 384)

    def __make_inception_B(self):
        layers = []
        if self.version == 'v4':
            for _ in range(7):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256, 192, 192, 224, 224, 256))
        elif self.version == 'res1':
            for _ in range(10):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))
        else:
            for _ in range(10):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        if self.version == 'v4':
            return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)
        elif self.version == 'res1':
            return Reduction_B_Res(896, 256, 384, 256, 256, 256, 256, 256)
        else:
            return Reduction_B_Res(1152, 256, 384, 256, 288, 256, 288, 320)

    def __make_inception_C(self):
        layers = []
        if self.version == 'v4':
            for _ in range(3):
                layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == 'res1':
            for _ in range(5):
                layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(5):
                layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        None
        out = self.fc(out)
        return out


class SeqConv(nn.Module):

    def __init__(self, in_chnls, out_chnls, kernel_size, activation=True):
        super(SeqConv, self).__init__()
        self.DWConv = BN_Conv2d(in_chnls, in_chnls, kernel_size, stride=1, padding=kernel_size // 2, groups=in_chnls, activation=activation)
        self.trans = BN_Conv2d(in_chnls, out_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        out = self.DWConv(x)
        return self.trans(out)


class MBConv(nn.Module):
    """Mobile inverted bottleneck conv"""

    def __init__(self, in_chnls, out_chnls, kernel_size, expansion, stride, is_se=False, activation=True):
        super(MBConv, self).__init__()
        self.is_se = is_se
        self.is_shortcut = stride == 1 and in_chnls == out_chnls
        self.trans1 = BN_Conv2d(in_chnls, in_chnls * expansion, 1, 1, 0, activation=activation)
        self.DWConv = BN_Conv2d(in_chnls * expansion, in_chnls * expansion, kernel_size, stride=stride, padding=kernel_size // 2, groups=in_chnls * expansion, activation=activation)
        if self.is_se:
            self.se = SE(in_chnls * expansion, 4)
        self.trans2 = BN_Conv2d(in_chnls * expansion, out_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        out = self.trans1(x)
        out = self.DWConv(out)
        if self.is_se:
            coeff = self.se(out)
            out *= coeff
        out = self.trans2(out)
        if self.is_shortcut:
            out += x
        return out


class MnasNet_A1(nn.Module):
    """MnasNet-A1"""
    _defaults = {'blocks': [2, 3, 4, 2, 3, 1], 'chnls': [24, 40, 80, 112, 160, 320], 'expans': [6, 3, 6, 6, 6, 6], 'k_sizes': [3, 5, 3, 3, 5, 3], 'strides': [2, 2, 2, 1, 2, 1], 'is_se': [False, True, False, True, True, False], 'dropout_ratio': 0.2}

    def __init__(self, num_classes=1000, input_size=224):
        super(MnasNet_A1, self).__init__()
        self.__dict__.update(self._defaults)
        self.body = self.__make_body()
        self.trans = BN_Conv2d(self.chnls[-1], 1280, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(self.dropout_ratio), nn.Linear(1280, num_classes))

    def __make_block(self, id):
        in_chnls = 16 if id == 0 else self.chnls[id - 1]
        strides = [self.strides[id]] + [1] * (self.blocks[id] - 1)
        layers = []
        for i in range(self.blocks[id]):
            layers.append(MBConv(in_chnls, self.chnls[id], self.k_sizes[id], self.expans[id], strides[i], self.is_se[id]))
            in_chnls = self.chnls[id]
        return nn.Sequential(*layers)

    def __make_body(self):
        blocks = [BN_Conv2d(3, 32, 3, 2, 1, activation=False), SeqConv(32, 16, 3)]
        for index in range(len(self.blocks)):
            blocks.append(self.__make_block(index))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.body(x)
        out = self.trans(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        None
        out = self.fc(out)
        return out


class ResNet(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes, is_se=False) ->object:
        super(ResNet, self).__init__()
        self.channels = 64
        self.block = block
        self.is_se = is_se
        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 512 if self.block.message == 'basic' else 512 * 4
        self.fc = nn.Linear(patches, num_classes)

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str('block_%d_%d' % (index, i))
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i], self.is_se))
            self.channels = channels if self.block.message == 'basic' else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes, is_se=False) ->object:
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(self.channels, num_classes)

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride, self.is_se))
            self.channels = self.cardinality * d * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ShuffleNet_v2(nn.Module):
    """ShuffleNet-v2"""
    _defaults = {'sets': {0.5, 1, 1.5, 2}, 'units': [3, 7, 3], 'chnl_sets': {(0.5): [24, 48, 96, 192, 1024], (1): [24, 116, 232, 464, 1024], (1.5): [24, 176, 352, 704, 1024], (2): [24, 244, 488, 976, 2048]}}

    def __init__(self, scale, num_cls, is_se=False, is_res=False) ->object:
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        assert scale in self.sets
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
        self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
        self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.body = self.__make_body()
        self.fc = nn.Linear(self.chnls[4], num_cls)

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls), BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units - 1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def __make_body(self):
        return nn.Sequential(self.conv1, self.maxpool, self.stage2, self.stage3, self.stage4, self.conv5, self.globalpool)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch: object, num_classes=1000) ->object:
        super(VGG, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return self.fc3(out)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BN_Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BN_Conv2d_Leaky,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BN_Conv_Mish,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Block_bank,
     lambda: ([], {'block_type': 4, 'in_channels': 4, 'b1_reduce': 4, 'b1': 4, 'b2_reduce': 4, 'b2': 4, 'b3': 4, 'b4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CSP_DenseBlock,
     lambda: ([], {'in_channels': 4, 'num_layers': 1, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DPN_Block,
     lambda: ([], {'in_chnls': 4, 'add_chnl': 4, 'cat_chnl': 4, 'cardinality': 4, 'd': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNet,
     lambda: ([], {'layers': [4, 4, 4, 4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Dark_block,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseBlock,
     lambda: ([], {'input_channels': 4, 'num_layers': 1, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GoogleNet,
     lambda: ([], {'str_version': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (Inception,
     lambda: ([], {'version': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     True),
    (Inception_A,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2': 4, 'b3_n1': 4, 'b3_n3': 4, 'b4_n1': 4, 'b4_n3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_A_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n3_1': 4, 'b3_n3_2': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_B,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2': 4, 'b3_n1': 4, 'b3_n1x7': 4, 'b3_n7x1': 4, 'b4_n1': 4, 'b4_n1x7_1': 4, 'b4_n7x1_1': 4, 'b4_n1x7_2': 4, 'b4_n7x1_2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_B_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n1x7': 4, 'b2_n7x1': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_C,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2': 4, 'b3_n1': 4, 'b3_n1x3_3x1': 4, 'b4_n1': 4, 'b4_n1x3': 4, 'b4_n3x1': 4, 'b4_n1x3_3x1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_C_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n1x3': 4, 'b2_n3x1': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_v3,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reduction_A,
     lambda: ([], {'in_channels': 4, 'k': 4, 'l': 4, 'm': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reduction_B_Res,
     lambda: ([], {'in_channels': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n3': 4, 'b4_n1': 4, 'b4_n3_1': 4, 'b4_n3_2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reduction_B_v4,
     lambda: ([], {'in_channels': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n1x7': 4, 'b3_n7x1': 4, 'b3_n3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNeXt,
     lambda: ([], {'layers': [4, 4, 4, 4], 'cardinality': 4, 'group_depth': 1, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (ResNeXt_Block,
     lambda: ([], {'in_chnls': 4, 'cardinality': 4, 'group_depth': 1, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'cardinality': 4, 'group_width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SE,
     lambda: ([], {'in_chnls': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stem,
     lambda: ([], {'in_channels': 4, 'num_blocks': 4, 'cardinality': 4, 'group_with': 4}),
     lambda: ([torch.rand([4, 16, 64, 64])], {}),
     True),
    (Stem_Res1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Stem_v4_Res2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_Cai_Yichao_torch_backbones(_paritybench_base):
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

