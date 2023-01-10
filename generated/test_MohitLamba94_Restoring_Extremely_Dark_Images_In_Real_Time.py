import sys
_module = sys.modules[__name__]
del sys
common_classes = _module
demo = _module
network = _module
time_complexity = _module
common_classes = _module
network = _module
train = _module
common = _module
common_classes = _module
network = _module
PixelUnShuffle = _module
common_classes = _module
network = _module
network_module = _module
common_classes = _module
network = _module
vainF_ssim = _module

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


import random


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.utils import weight_norm as wn


import torch.nn as nn


import time


from torchvision import models


import torch.optim as optim


import math


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn import Parameter


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.squeeze = SEBlock(n_feats)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def ICNR(tensor, upscale_factor=2, negative_slope=1, fan_type='fan_in'):
    new_shape = [int(tensor.shape[0] / upscale_factor ** 2)] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    nn.init.kaiming_normal_(subkernel, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)
    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=4, num_classes=3, weight_normalization=True):
    layers = []
    if bn:
        m = nn.BatchNorm2d(inc)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)
    if activation == 'before':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
    if pixelshuffle_init:
        m = nn.Conv2d(in_channels=inc, out_channels=num_classes * upscale ** 2, kernel_size=3, padding=3 // 2, groups=1, bias=True, stride=1)
        nn.init.constant_(m.bias, 0)
        with torch.no_grad():
            kernel = ICNR(m.weight, upscale, negative_slope, fan_type)
            m.weight.copy_(kernel)
    else:
        m = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=groups, bias=bias, stride=1)
        init_gain = 0.02
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight, gain=init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)
    if activation == 'after':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
    return nn.Sequential(*layers)


class make_dense(nn.Module):

    def __init__(self, nChannels=64, growthRate=32, pos=False):
        super(make_dense, self).__init__()
        kernel_size = 3
        if pos == 'first':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization=True)
        elif pos == 'middle':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization=True)
        elif pos == 'last':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization=True)
        else:
            raise NotImplementedError('ReLU position error in make_dense')

    def forward(self, x):
        return torch.cat((x, self.conv(x)), 1)


class RDB(nn.Module):

    def __init__(self, nChannels=96, nDenselayer=5, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        modules.append(make_dense(nChannels_, growthRate, 'first'))
        nChannels_ += growthRate
        for i in range(nDenselayer - 2):
            modules.append(make_dense(nChannels_, growthRate, 'middle'))
            nChannels_ += growthRate
        modules.append(make_dense(nChannels_, growthRate, 'last'))
        nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = conv_layer(nChannels_, nChannels, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization=True)

    def forward(self, x):
        return self.conv_1x1(self.dense_layers(x)) + x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.up2 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(self.downshuffle(x, 2)))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        conv10 = self.conv10_1(conv9)
        out = self.up2(conv10)
        return out

    def downshuffle(self, var, r):
        b, c, h, w = var.size()
        out_channel = c * r ** 2
        out_h = h // r
        out_w = w // r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class SID(nn.Module):

    def __init__(self):
        super(SID, self).__init__()
        self.up2 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(self.downshuffle(x, 2)))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        conv10 = self.conv10_1(conv9)
        out = self.up2(conv10)
        return out

    def downshuffle(self, var, r):
        b, c, h, w = var.size()
        out_channel = c * r ** 2
        out_h = h // r
        out_w = w // r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = torch.nn.L1Loss()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].eval())
        for name, param in self.named_parameters():
            param.requires_grad = False
            None

    def VGGfeatures(self, x):
        x = self.slice1(x)
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        return relu2_2, relu3_3

    def forward(self, ip, target, which='relu2'):
        ip = (ip - self.mean) / self.std
        target = (target - self.mean) / self.std
        ip_relu2_2, ip_relu3_3 = self.VGGfeatures(ip)
        target_relu2_2, target_relu3_3 = self.VGGfeatures(target)
        if which == 'relu2':
            loss = self.l1(ip_relu2_2, target_relu2_2)
        elif which == 'relu3':
            loss = self.l1(ip_relu3_3, target_relu3_3)
        elif which == 'both':
            loss = self.l1(ip_relu2_2, target_relu2_2) + self.l1(ip_relu3_3, target_relu3_3)
        else:
            raise NotImplementedError('Incorrect WHICH in perceptual loss.')
        return loss


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU())
                elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]
    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor], device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnShuffle(nn.Module):

    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """
        return pixel_unshuffle(input, self.downscale_factor)


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-08, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Conv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero', activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResConv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero', activation='lrelu', norm='none', sn=False, scale_factor=2):
        super(ResConv2dLayer, self).__init__()
        self.conv2d = nn.Sequential(Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn), Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn), Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation='none', norm=norm, sn=sn))
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = out + residual
        if self.activation:
            out = self.activation(out)
        return out


class SGN(nn.Module):

    def __init__(self, opt):
        super(SGN, self).__init__()
        self.top1 = Conv2dLayer(opt.in_channels * 4 ** 3, opt.start_channels * 2 ** 3, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.top2 = ResConv2dLayer(opt.start_channels * 2 ** 3, opt.start_channels * 2 ** 3, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.top3 = Conv2dLayer(opt.start_channels * 2 ** 3, opt.start_channels * 2 ** 3, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.mid1 = Conv2dLayer(opt.in_channels * 4 ** 2, opt.start_channels * 2 ** 2, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.mid2 = Conv2dLayer(int(opt.start_channels * (2 ** 2 + 2 ** 3 / 4)), opt.start_channels * 2 ** 2, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.mid3 = ResConv2dLayer(opt.start_channels * 2 ** 2, opt.start_channels * 2 ** 2, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.mid4 = Conv2dLayer(opt.start_channels * 2 ** 2, opt.start_channels * 2 ** 2, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.bot1 = Conv2dLayer(opt.in_channels * 4 ** 1, opt.start_channels * 2 ** 1, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.bot2 = Conv2dLayer(int(opt.start_channels * (2 ** 1 + 2 ** 2 / 4)), opt.start_channels * 2 ** 1, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.bot3 = ResConv2dLayer(opt.start_channels * 2 ** 1, opt.start_channels * 2 ** 1, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.bot4 = Conv2dLayer(opt.start_channels * 2 ** 1, opt.start_channels * 2 ** 1, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.main1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.main2 = Conv2dLayer(int(opt.start_channels * (2 ** 0 + 2 ** 1 / 4)), opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        self.main3 = nn.ModuleList([Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)])
        self.main3.append(Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm))
        self.main3.append(Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm))
        for i in range(opt.m_block):
            self.main3.append(Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm))
        self.main4 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)

    def forward(self, x):
        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)
        x2 = PixelUnShuffle.pixel_unshuffle(x, 4)
        x3 = PixelUnShuffle.pixel_unshuffle(x, 8)
        x3 = self.top1(x3)
        x3 = self.top2(x3)
        x3 = self.top3(x3)
        x3 = F.pixel_shuffle(x3, 2)
        x2 = self.mid1(x2)
        x2 = torch.cat((x2, x3), 1)
        x2 = self.mid2(x2)
        x2 = self.mid3(x2)
        x2 = self.mid4(x2)
        x2 = F.pixel_shuffle(x2, 2)
        x1 = self.bot1(x1)
        x1 = torch.cat((x1, x2), 1)
        x1 = self.bot2(x1)
        x1 = self.bot3(x1)
        x1 = self.bot4(x1)
        x1 = F.pixel_shuffle(x1, 2)
        x = self.main1(x)
        x = torch.cat((x, x1), 1)
        x = self.main2(x)
        for model in self.main3:
            x = model(x)
        x = self.main4(x)
        return x


def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size)
    coords -= size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """ Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03), nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')
    if win is not None:
        win_size = win.shape[-1]
    if not win_size % 2 == 1:
        raise ValueError('Window size should be odd.')
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


class SSIM(torch.nn.Module):

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


def ms_ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')
    if win is not None:
        win_size = win.shape[-1]
    if not win_size % 2 == 1:
        raise ValueError('Window size should be odd.')
    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * 2 ** 4, 'Image size should be larger than %d due to the 4 downsamplings in ms-ssim' % ((win_size - 1) * 2 ** 4)
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights)
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = X.shape[2] % 2, X.shape[3] % 2
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class MS_SSIM(torch.nn.Module):

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, weights=None, K=(0.01, 0.03)):
        """ class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, weights=self.weights, K=self.K)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'conv': _mock_layer, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanShift,
     lambda: ([], {'rgb_range': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (RDB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     False),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feats': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResConv2dLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 10, 10])], {}),
     False),
    (SEBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SID,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
]

class Test_MohitLamba94_Restoring_Extremely_Dark_Images_In_Real_Time(_paritybench_base):
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

