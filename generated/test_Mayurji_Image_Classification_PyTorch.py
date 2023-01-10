import sys
_module = sys.modules[__name__]
del sys
AlexNet = _module
BatchNorm_ConvNet = _module
Darknet53 = _module
DenseNet = _module
EfficientNet = _module
EfficientNetV2 = _module
GoogLeNet = _module
InceptionV3 = _module
MLPMixer = _module
MobileNetV1 = _module
MobileNetV2 = _module
NiN = _module
ResMLP = _module
ResNeXt = _module
ResNet = _module
SAM = _module
SENet = _module
ShuffleNet = _module
SqueezeNet = _module
VGG = _module
ViT = _module
Xception = _module
augmentations = _module
convNet = _module
dataset = _module
gMLP = _module
main = _module
optimizer = _module
plot = _module
trainAndTestWithSAM = _module
train_test = _module

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


import torch


from torch.nn.modules import batchnorm


import torchvision


import torchvision.transforms as transforms


import matplotlib.pyplot as plt


import time


from torch import nn


import math


from torch.nn import functional as F


import torch.nn.functional as F


from torch.autograd import Variable


from functools import partial


import numpy as np


from torch import einsum


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


class AlexNet(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(4096, n_classes))
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        return out


def batchNormalization(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    X - dataset
    gamma - scale parameter
    beta - shift parameter
    moving_mean - used during inference 
    moving_var - used during inference
    """
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):

    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = 1, num_features
        else:
            shape = 1, num_features, 1, 1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean
            self.moving_var = self.moving_var
        Y, self.moving_mean, self.moving_var = batchNormalization(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-05, momentum=0.9)
        return Y


class LeNet_BN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, bias=False), BatchNorm(6, num_dims=4), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2), nn.Conv2d(6, 16, kernel_size=5, bias=False), BatchNorm(16, num_dims=4), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(16 * 4 * 4, 120, bias=False), BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.Linear(120, 84, bias=False), BatchNorm(84, num_dims=2), nn.Sigmoid(), nn.Linear(84, 10, bias=False))

    def forward(self, X):
        out = self.conv(X)
        out = self.fc(out)
        return out


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_num), nn.LeakyReLU())


class DarkResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(in_channels / 2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):

    def __init__(self, input_channel, n_classes):
        super(Darknet53, self).__init__()
        self.conv1 = conv_batch(input_channel, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block=DarkResidualBlock, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block=DarkResidualBlock, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block=DarkResidualBlock, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block=DarkResidualBlock, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block=DarkResidualBlock, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

    def make_layer(self, in_channels, num_blocks, block):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


class Shuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):

    def __init__(self, input_channel, output_channel, stride, groups):
        super().__init__()
        self.stride = stride
        in_between_channel = int(output_channel / 4)
        g = 1 if input_channel == 24 else groups
        self.conv1x1_1 = nn.Sequential(nn.Conv2d(input_channel, in_between_channel, kernel_size=1, groups=g, bias=False), nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.shuffle = Shuffle(groups=g)
        self.conv1x1_2 = nn.Sequential(nn.Conv2d(in_between_channel, in_between_channel, kernel_size=3, stride=stride, padding=1, groups=in_between_channel, bias=False), nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.conv1x1_3 = nn.Sequential(nn.Conv2d(in_between_channel, output_channel, kernel_size=1, groups=groups, bias=False), nn.BatchNorm2d(output_channel))
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.shuffle(out)
        out = self.conv1x1_2(out)
        out = self.conv1x1_3(out)
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, input_channel, growthRate, depth, reduction, n_classes, bottleneck):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(input_channel, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn1(out))
        out = torch.squeeze(F.adaptive_avg_pool2d(out, 1))
        out = F.log_softmax(self.fc(out))
        return out


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SqueezeAndExcitation(nn.Module):

    def __init__(self, channel, squeeze_channel, se_ratio):
        super().__init__()
        squeeze_channel = squeeze_channel * se_ratio
        if not squeeze_channel.is_integer():
            raise ValueError('channels must be divisible by 1/se_ratio')
        squeeze_channel = int(squeeze_channel)
        self.se_reduce = nn.Conv2d(channel, squeeze_channel, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_excite = nn.Conv2d(squeeze_channel, channel, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear1(self.se_excite(y))
        y = x * y
        return y


def batchNorm(channels, eps=0.001, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)


def dropPath(x, drop_probability, training):
    if drop_probability > 0 and training:
        keep_probability = 1 - drop_probability
        if x.is_cuda:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))
        x.div_(keep_probability)
        x.mul_(mask)
    return x


class MBConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super().__init__()
        expand = expand_ratio != 1
        expand_channel = in_channel * expand_ratio
        se = se_ratio != 0
        self.residual_connection = stride == 1 and in_channel == out_channel
        self.drop_path_rate = drop_path_rate
        conv = []
        if expand:
            pw_expansion = nn.Sequential(nn.Conv2d(in_channel, expand_channel, 1, 1, 0, bias=False), batchNorm(expand_channel), Swish())
            conv.append(pw_expansion)
        dw = nn.Sequential(nn.Conv2d(expand_channel, expand_channel, kernel_size, stride, kernel_size // 2, groups=expand_channel, bias=False), batchNorm(expand_channel), Swish())
        conv.append(dw)
        if se:
            squeeze_excite = SqueezeAndExcitation(expand_channel, in_channel, se_ratio)
            conv.append(squeeze_excite)
        pw_projection = nn.Sequential(nn.Conv2d(expand_channel, out_channel, 1, 1, 0, bias=False), batchNorm(out_channel))
        conv.append(pw_projection)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + dropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)


def conv1x1(input_channel, output_channel):
    return nn.Sequential(nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channel), nn.ReLU6(inplace=True))


def conv3x3(input_channel, output_channel, stride):
    return nn.Sequential(nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False), nn.BatchNorm2d(output_channel), nn.ReLU6(inplace=True))


def roundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


def roundRepeats(r):
    return int(math.ceil(r))


class EfficientNet(nn.Module):
    cfg = [[32, 16, 3, 1, 1, 0.25, 1], [16, 24, 3, 2, 6, 0.25, 2], [24, 40, 5, 2, 6, 0.25, 2], [40, 80, 3, 2, 6, 0.25, 3], [80, 112, 5, 1, 6, 0.25, 3], [112, 192, 5, 2, 6, 0.25, 4], [192, 320, 3, 1, 6, 0.25, 1]]

    def __init__(self, input_channels, param, n_classes, stem_channels=32, feature_size=1280, drop_connect_rate=0.2):
        super().__init__()
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = roundChannels(stem_channels * width_coefficient)
            for conf in self.cfg:
                conf[0] = roundChannels(conf[0] * width_coefficient)
                conf[1] = roundChannels(conf[1] * width_coefficient)
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.cfg:
                conf[6] = roundRepeats(conf[6] * depth_coefficient)
        input_size = param[2]
        self.stem_conv = conv3x3(input_channels, stem_channels, 2)
        total_blocks = 0
        for conf in self.cfg:
            total_blocks += conf[6]
        blocks = []
        for in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, repeats in self.cfg:
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats - 1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channel, out_channel, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)
        self.head_conv = conv1x1(self.cfg[-1][1], feature_size)
        self.dropout = nn.Dropout(param[3])
        self.classifier = nn.Linear(feature_size, n_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
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


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class SELayer(nn.Module):

    def __init__(self, input, output, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(output, make_divisible(input // reduction, 8)), SiLU(), nn.Linear(make_divisible(input // reduction, 8), output), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):

    def __init__(self, input, output, stride, expand_ratio, use_se) ->None:
        super().__init__()
        assert stride in [1, 2]
        hidden_dimension = round(input * expand_ratio)
        self.identity = stride == 1 and input == output
        if use_se:
            self.conv = nn.Sequential(nn.Conv2d(input, hidden_dimension, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dimension), SiLU(), nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False), nn.BatchNorm2d(hidden_dimension), SiLU(), SELayer(input, hidden_dimension), nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False), nn.BatchNorm2d(output))
        else:
            self.conv = nn.Sequential(nn.Conv2d(input, hidden_dimension, 3, stride, 1, bias=False), nn.BatchNorm2d(hidden_dimension), SiLU(), nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False), nn.BatchNorm2d(output))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv1x1_bn(input, output):
    return nn.Sequential(nn.Conv2d(input, output, 1, 1, 0, bias=False), nn.BatchNorm2d(output), SiLU())


def conv3x3_bn(input, output, stride):
    return nn.Sequential(nn.Conv2d(input, output, 3, stride, 1, bias=False), nn.BatchNorm2d(output), SiLU())


class EfficientNetV2(nn.Module):

    def __init__(self, cfgs, in_channel, num_classes=10, width_multiplier=1.0) ->None:
        super().__init__()
        self.cfgs = cfgs
        input_channel = make_divisible(24 * width_multiplier, 8)
        layers = [conv3x3_bn(in_channel, input_channel, 2)]
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = make_divisible(c * width_multiplier, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = make_divisible(1792 * width_multiplier, 8) if width_multiplier > 1.0 else 1792
        self.conv = conv1x1_bn(input_channel, output_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
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
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class Inception(nn.Module):

    def __init__(self, in_channel, out_channel_1, out_channel_2, out_channel_3, out_channel_4, **kwargs):
        super().__init__()
        self.p1_1 = nn.Conv2d(in_channel, out_channel_1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channel, out_channel_2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(out_channel_2[0], out_channel_2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channel, out_channel_3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(out_channel_3[0], out_channel_3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channel, out_channel_4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(), nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        """
		Explaining Only two Inception block, there are totally 9 Inception Block.

		First Inception Module in b3, 

		in_channel=192, 
		out_channels = {	
			self.p1_1: 64, 
			self.p2_1: 96, self.p2_2: 128, 
			self.p3_1: 16, self.p3_2: 32,
			self.p4_1: 32
			}
		Add all out_channel => 64 + 128 + 32 + 32 = 256, which is our input to next Inception module.

		For each parallel block, the input is 192, and we can see in second and third path we reduce in=192 to out=96 
		and in=192 to out=16 respectively.
		
		Second Inception Module

		in_channel=256, 
		out_channels = {	
			self.p1_1: 128, 
			self.p2_1: 128, self.p2_2: 192, 
			self.p3_1: 32, self.p3_2: 96,
			self.p4_1: 64
			}
		Add all out_channel => 128 + 192 + 96 + 64 = 480, which is our input to next Inception module.

		For each parallel block, the input is 256, and we can see in second and third path we reduce in=256 to out=128 
		and in=256 to out=32 respectively.
		"""
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32), Inception(256, 128, (128, 192), (32, 96), 64), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64), Inception(512, 160, (112, 224), (24, 64), 64), Inception(512, 128, (128, 256), (24, 64), 64), Inception(512, 112, (144, 288), (32, 64), 64), Inception(528, 256, (160, 320), (32, 128), 128), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128), Inception(832, 384, (192, 384), (48, 128), 128), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Linear(1024, n_classes)
        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.fc(out)
        return out


class BasicConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channel, output_channel, bias=False, **kwargs), nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))

    def forward(self, X):
        return self.conv(X)


class InceptionA(nn.Module):

    def __init__(self, input_channel, pool_features):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channel, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(BasicConvBlock(input_channel, 48, kernel_size=1), BasicConvBlock(48, 64, kernel_size=5, padding=2))
        self.branch3x3 = nn.Sequential(BasicConvBlock(input_channel, 64, kernel_size=1), BasicConvBlock(64, 96, kernel_size=3, padding=1), BasicConvBlock(96, 96, kernel_size=3, padding=1))
        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConvBlock(input_channel, pool_features, kernel_size=3, padding=1))

    def forward(self, X):
        branch1x1 = self.branch1x1(X)
        branch5x5 = self.branch5x5(X)
        branch3x3 = self.branch3x3(X)
        branchpool = self.branchpool(X)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = BasicConvBlock(input_channels, 384, kernel_size=3, stride=2)
        self.branch3x3stack = nn.Sequential(BasicConvBlock(input_channels, 64, kernel_size=1), BasicConvBlock(64, 96, kernel_size=3, padding=1), BasicConvBlock(96, 96, kernel_size=3, stride=2))
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, X):
        branch3x3 = self.branch3x3(X)
        branch3x3stack = self.branch3x3stack(X)
        branchpool = self.branchpool(X)
        """We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(BasicConvBlock(input_channels, c7, kernel_size=1), BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)), BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.branch7x7stack = nn.Sequential(BasicConvBlock(input_channels, c7, kernel_size=1), BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)), BasicConvBlock(c7, c7, kernel_size=(1, 7), padding=(0, 3)), BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)), BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConvBlock(input_channels, 192, kernel_size=1))

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
        self.branch3x3 = nn.Sequential(BasicConvBlock(input_channels, 192, kernel_size=1), BasicConvBlock(192, 320, kernel_size=3, stride=2))
        self.branch7x7 = nn.Sequential(BasicConvBlock(input_channels, 192, kernel_size=1), BasicConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)), BasicConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0)), BasicConvBlock(192, 192, kernel_size=3, stride=2))
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
        self.branch1x1 = BasicConvBlock(input_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConvBlock(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3stack_1 = BasicConvBlock(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConvBlock(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), BasicConvBlock(input_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [self.branch3x3stack_3a(branch3x3stack), self.branch3x3stack_3b(branch3x3stack)]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):

    def __init__(self, input_channel, n_classes=10):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConvBlock(input_channel, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConvBlock(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConvBlock(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConvBlock(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConvBlock(80, 192, kernel_size=3)
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
        self.linear = nn.Linear(2048, n_classes)

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


class FeedForward(nn.Module):

    def __init__(self, input_channel, output_channel, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_channel, output_channel), nn.ReLU(), nn.Dropout(dropout), nn.Linear(output_channel, input_channel), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.token_mix = nn.Sequential(nn.LayerNorm(dim), Rearrange('b p c -> b c p'), FeedForward(num_patch, token_dim, dropout), Rearrange('b c p -> b p c'))
        self.channel_mix = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, channel_dim, dropout))

    def forward(self, x):
        px = x + self.token_mix(x)
        cx = px + self.channel_mix(px)
        return cx


class MLPMixer(nn.Module):

    def __init__(self, input_channels, dim, n_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(nn.Conv2d(input_channels, dim, patch_size, patch_size), Rearrange('b c h w -> b (h w) c'))
        self.mixer_blocks = nn.ModuleList([])
        for i in range(depth):
            cx = MixerBlock(dim, self.num_patch, token_dim, channel_dim)
            self.mixer_blocks.append(cx)
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.Linear(dim, n_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)


class MobileNetV1(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(input_channel, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, n_classes)
        self.model.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=0.001)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, input_channel, out_channel, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2], 'Stride value is greater than 2'
        hidden_dimension = round(input_channel * expand_ratio)
        self.identity = stride == 1 and input_channel == out_channel
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False), nn.BatchNorm2d(hidden_dimension), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dimension, out_channel, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channel))
        else:
            self.conv = nn.Sequential(nn.Conv2d(input_channel, hidden_dimension, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dimension), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False), nn.BatchNorm2d(hidden_dimension), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dimension, out_channel, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, input_channel, n_classes=10, width_multipler=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        first_channel = 32
        last_channel = 1280
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        self.last_channel = make_divisible(last_channel * width_multipler) if width_multipler > 1.0 else last_channel
        self.features = [conv3x3(input_channel, first_channel, 2)]
        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_multipler) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(first_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(first_channel, output_channel, 1, expand_ratio=t))
                first_channel = output_channel
        self.features.append(conv1x1(first_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.last_channel, n_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class NIN(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()

        def NINBlock(input_channel, out_channel, kernel_size, strides, padding):
            return nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding), nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1), nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1), nn.ReLU())
        self.layers = nn.Sequential(NINBlock(input_channel, 96, kernel_size=11, strides=4, padding=0), nn.MaxPool2d(3, stride=2), NINBlock(96, 256, kernel_size=5, strides=1, padding=2), nn.MaxPool2d(3, stride=2), NINBlock(256, 384, kernel_size=3, strides=1, padding=1), nn.MaxPool2d(3, stride=2), nn.Dropout(0.5), NINBlock(384, n_classes, kernel_size=3, strides=1, padding=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.layers(x)
        return out


class Aff(nn.Module):
    """
    Affine Transformation
    """

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class MLPBlock(nn.Module):

    def __init__(self, dim, num_patch, mlp_dim, dropout=0.0, init_values=0.0001):
        super().__init__()
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(Rearrange('b n d -> b d n'), nn.Linear(num_patch, num_patch), Rearrange('b d n -> b n d'))
        self.ff = nn.Sequential(FeedForward(dim, mlp_dim, dropout))
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x


class ResMLP(nn.Module):

    def __init__(self, in_channels, dim, n_classes, patch_size, image_size, depth, mlp_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(nn.Conv2d(in_channels, dim, patch_size, patch_size), Rearrange('b c h w -> b (h w) c'))
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(MLPBlock(dim, self.num_patch, mlp_dim))
        self.affine = Aff(dim)
        self.mlp_head = nn.Sequential(nn.Linear(dim, n_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)


class SeparableConv(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.dwc = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding, dilation, groups=input_channel, bias=bias), nn.Conv2d(input_channel, output_channel, 1, 1, 0, 1, 1, bias=bias))

    def forward(self, X):
        return self.dwc(X)


class Block(nn.Module):

    def __init__(self, input_channel, out_channel, reps, strides=1, relu=True, grow_first=True):
        super().__init__()
        if out_channel != input_channel or strides != 1:
            self.skipConnection = nn.Sequential(nn.Conv2d(input_channel, out_channel, 1, stride=strides, bias=False), nn.BatchNorm2d(out_channel))
        else:
            self.skipConnection = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = input_channel
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channel))
            filters = out_channel
        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv(input_channel, out_channel, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channel))
        if not relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        X = self.rep(input)
        if self.skipConnection:
            skip = self.skipConnection(input)
        else:
            skip = input
        X += skip
        return X


class ResNeXt(nn.Module):

    def __init__(self, input_channel, num_blocks, cardinality, bottleneck_width, n_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, n_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Residual(nn.Module):

    def __init__(self, in_channel, out_channel, use_1x1Conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if use_1x1Conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            X = self.conv3(X)
        out += X
        return F.relu(out)


def residualBlock(in_channel, out_channel, num_residuals, first_block=False):
    blks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blks.append(Residual(in_channel, out_channel, use_1x1Conv=True, strides=2))
        else:
            blks.append(Residual(out_channel, out_channel))
    return blks


class ResNet(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b4 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b5 = nn.Sequential(*residualBlock(256, 512, 2))
        self.finalLayer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, n_classes))
        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=0.001)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.finalLayer(out)
        return out


class SEBlock(nn.Module):

    def __init__(self, C, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(C, C // r, bias=False), nn.ReLU(), nn.Linear(C // r, C, bias=False), nn.Sigmoid())

    def forward(self, x):
        bs, c, _, _ = x.shape
        s = self.squeeze(x).view(bs, c)
        e = self.excitation(s).view(bs, c, 1, 1)
        return x * e.expand_as(x)


class SENet(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*[SEBlock(C=64)])
        self.b3 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b4 = nn.Sequential(*[SEBlock(C=64)])
        self.b5 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b6 = nn.Sequential(*[SEBlock(C=128)])
        self.b7 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b8 = nn.Sequential(*[SEBlock(C=256)])
        self.b9 = nn.Sequential(*residualBlock(256, 512, 2))
        self.b10 = nn.Sequential(*[SEBlock(C=512)])
        self.finalLayer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, n_classes))
        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.b6.apply(self.init_weights)
        self.b7.apply(self.init_weights)
        self.b8.apply(self.init_weights)
        self.b9.apply(self.init_weights)
        self.b10.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=0.001)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.b6(out)
        out = self.b7(out)
        out = self.b8(out)
        out = self.b9(out)
        out = self.finalLayer(out)
        return out


class ShuffleNet(nn.Module):

    def __init__(self, cfg, input_channel, n_classes):
        super().__init__()
        output_channels = cfg['out']
        n_blocks = cfg['n_blocks']
        groups = cfg['groups']
        self.in_channels = 24
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, 24, kernel_size=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self.layer1 = self.make_layer(output_channels[0], n_blocks[0], groups)
        self.layer2 = self.make_layer(output_channels[1], n_blocks[1], groups)
        self.layer3 = self.make_layer(output_channels[2], n_blocks[2], groups)
        self.linear = nn.Linear(output_channels[2], n_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def make_layer(self, out_channel, n_blocks, groups):
        layers = []
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            cat_channels = self.in_channels if i == 0 else 0
            layers.append(Bottleneck(self.in_channels, out_channel - cat_channels, stride=stride, groups=groups))
            self.in_channels = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Fire(nn.Module):

    def __init__(self, in_channel, squeeze_channel, expand_channel):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channel, squeeze_channel, kernel_size=1, stride=1), nn.BatchNorm2d(squeeze_channel), nn.ReLU(inplace=True))
        self.expand1x1 = nn.Sequential(nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1, stride=1), nn.BatchNorm2d(expand_channel))
        self.expand3x3 = nn.Sequential(nn.Conv2d(squeeze_channel, expand_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(expand_channel))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        squeezeOut = self.squeeze(x)
        expandOut_1x1 = self.expand1x1(squeezeOut)
        expandOut_3x3 = self.expand3x3(squeezeOut)
        output = torch.cat([expandOut_1x1, expandOut_3x3], 1)
        output = F.relu(output)
        return output


class SqueezeNet(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channel, 96, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.Fire2 = Fire(96, 16, 64)
        self.Fire3 = Fire(128, 16, 64)
        self.Fire4 = Fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Fire5 = Fire(256, 32, 128)
        self.Fire6 = Fire(256, 48, 192)
        self.Fire7 = Fire(384, 48, 192)
        self.Fire8 = Fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Fire9 = Fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.Fire2(x)
        x = self.Fire3(x)
        x = self.Fire4(x)
        x = self.maxpool2(x)
        x = self.Fire5(x)
        x = self.Fire6(x)
        x = self.Fire7(x)
        x = self.Fire8(x)
        x = self.maxpool3(x)
        x = self.Fire9(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        x = x.view(x.size(0), -1)
        return x


class VGG11(nn.Module):

    def __init__(self, input_channel, n_classes, image_resolution, VGGArchitecture=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        super().__init__()
        self.input_channel = input_channel

        def VGGBlock(num_convs, input_channel, output_channel):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                input_channel = output_channel
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)
        conv_blcks = []
        for num_convs, output_channel in VGGArchitecture:
            conv_blcks.append(VGGBlock(num_convs, self.input_channel, output_channel))
            self.input_channel = output_channel
        self.layers = nn.Sequential(*conv_blcks, nn.Flatten(), nn.Linear(output_channel * (image_resolution // 2 ** len(VGGArchitecture)) * (image_resolution // 2 ** len(VGGArchitecture)), 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, n_classes))
        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight)

    def forward(self, x):
        out = self.layers(x)
        return out


class PreNorm(nn.Module):

    def __init__(self, dim, func):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x, **kwargs):
        return self.func(self.norm(x), **kwargs)


class Attention(nn.Module):
    """
    Attention (Q, K, V) = softmax( Q . (K.T) / (d_m ** 0.5) ) . V
    """

    def __init__(self, input_channel, heads=8, dimension_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dimension_head * heads
        project_out = not (heads == 1 and dimension_head == input_channel)
        self.heads = heads
        self.scale = dimension_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_QKV = nn.Linear(input_channel, inner_dim * 3, bias=False)
        self.out = nn.Sequential(nn.Linear(inner_dim, input_channel), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        QKV = self.to_QKV(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), QKV)
        dots = einsum('b h i d, b h j d -> b h i j', Q, K) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)


class Transformers(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dimension_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, n_classes, dim, depth, heads, mlp_dim, pool='cls', input_channel=1, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimension must be divisible by patch size.'
        num_patches = image_height // patch_height * (image_width // patch_width)
        patch_dim = input_channel * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformers(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, input):
        """
        to_patch_embedding:
            input image: (batch x channel x height x width) => (32 x 3 x 224 x 224)
            Using rearrange: 
                p1, p2 = patch_size
                * (b x c x h x w) => (b x (h x p1) x (w x p2) x c)
                * (b x (h x p1) x (w x p2) x c) => (b x (h w) x (p1 x p2 x c))
                * (32 x (7 x 32) x (7 x 32) x 3) => (32 x 49 x 3072)
            
            Passing through Linear Layer:
                * (32 x 49 x 3072) => (32 x 49 x 1024)
            
        Adding Positional Embedding 
            * (32 x 49 x 1024) + (32 x 1 x 1024) => (32 x 50 x 1024)
        
        Transformers
        Input is (32 x 50 x 1024)
            * LayerNorm -> Attention => output1
                * (32 x 50 x 1024) ---LN---> (32 x 50 x 1024)
                * (32 x 50 x 1024) ---Attn--> 
                    * Using Linear Layer, we generate Q, K, V matrices by splitting 
                        * (32 x 50 x 1024) => (32 x 50 x 3072) using chunks
                        * (32 x 50 x 3072) => Q - (32 x 50 x 1024) 
                                              K - (32 x 50 x 1024) 
                                              V - (32 x 50 x 1024)
                        * Using Rearrange, 
                            Q - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                            K - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                            V - (32 x 50 x 1024) ==(1024 = 16 x 64)==> (32 x 16 x 50 x 64)
                        * Attention Weights, Q and K, dots operation
                            einsum(Q, K), 
                            (32 x 16 x 50 x 64) . (32 x 16 x 64 x 50) => (32 x 16 x 50 x 50)
                        * Attention Weights and Value
                            Attn . Value
                            (32 x 16 x 50 x 50) . (32 x 16 x 50 x 64) ==(16 x 64 = 1024)==> (32 x 50 x 1024)
                        * Attn.Value -> Linear
                            (32 x 50 x 1024) --Linear--> (32 x 50 x 1024)
                Attention output => (32 x 50 x 1024)
            
            * output => LayerNorm --> FeedForward
                * To LayerNorm, (32 x 50 x 1024) --LN--> (32 x 50 x 1024) from output of attention.
                * To Linear, (32 x 50 x 1024) --Linear--> (32 x 50 x 1024) from output of LayerNorm
                * (32 x 50 x 1024) <---Residual--> output(attention output)
                * (32 x 50 x 1024)
      MLP Head
      * Linear Layer to n_classes
          * (32 x 50 x 1024) => * (32 x 10)

        """
        x = self.to_patch_embedding(input)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class Xception(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU(inplace=True)
        self.initBlock = nn.Sequential(nn.Conv2d(input_channel, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.block1 = Block(64, 128, 2, 2, relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, relu=True, grow_first=False)
        self.conv3 = SeparableConv(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, self.n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.initBlock(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        out = h2.reshape(h2.size(0), -1)
        out = self.fc(out)
        return out


class SpatialGatingUnit(nn.Module):

    def __init__(self, d_ffn, seq_len) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_projection = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_projection.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_projection(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):

    def __init__(self, d_model, d_ffn, seq_len) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_projection_1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_projection_2 = nn.Linear(d_ffn, d_model)
        self.SGU = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_projection_1(x))
        x = self.SGU(x)
        x = self.channel_projection_2(x)
        out = x + residual
        return out


class gMLP(nn.Module):

    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6) ->None:
        super().__init__()
        self.model = nn.Sequential(*[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)])

    def forward(self, x):
        return self.model(x)


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "'image_size' should be divisible by patch size"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class gMLPForImageClassification(gMLP):

    def __init__(self, in_channels, n_classes, image_size, patch_size, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.patcher = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.classifier(embedding)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Aff,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConvBlock,
     lambda: ([], {'input_channel': 4, 'output_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm,
     lambda: ([], {'num_features': 4, 'num_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Block,
     lambda: ([], {'input_channel': 4, 'out_channel': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'input_channel': 4, 'output_channel': 4, 'stride': 1, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkResidualBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Darknet53,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fire,
     lambda: ([], {'in_channel': 4, 'squeeze_channel': 4, 'expand_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GoogLeNet,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionA,
     lambda: ([], {'input_channel': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionB,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'input_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionD,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionV3,
     lambda: ([], {'input_channel': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'input_channel': 4, 'out_channel': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV1,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {'input_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NIN,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 128, 128])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'func': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SEBlock,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SENet,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SeparableConv,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Shuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Xception,
     lambda: ([], {'input_channel': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (gMLPBlock,
     lambda: ([], {'d_model': 4, 'd_ffn': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (gMLPForImageClassification,
     lambda: ([], {'in_channels': 4, 'n_classes': 4, 'image_size': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_Mayurji_Image_Classification_PyTorch(_paritybench_base):
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

