import sys
_module = sys.modules[__name__]
del sys
x = _module
config = _module
GCC = _module
loading_data = _module
setting = _module
Mall = _module
loading_data = _module
QNRF = _module
loading_data = _module
SHHA = _module
loading_data = _module
SHHB = _module
loading_data = _module
UCF50 = _module
loading_data = _module
UCSD = _module
loading_data = _module
WE = _module
loading_data = _module
datasets = _module
misc = _module
cal_mean = _module
layer = _module
pytorch_ssim = _module
ssim_loss = _module
transforms = _module
utils = _module
CC = _module
M2T2OCC = _module
CMTL = _module
M2T2OCC_Model = _module
M2TCC = _module
SANet = _module
M2TCC_Model = _module
AlexNet = _module
CSRNet = _module
MCNN = _module
Res101 = _module
Res101_SFCN = _module
Res50 = _module
VGG = _module
VGG_decoder = _module
SCC_Model = _module
models = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
config = _module
test = _module
train = _module
trainer = _module
trainer_for_CMTL = _module
trainer_for_M2TCC = _module

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


import numpy as np


from scipy import io as sio


from torch.utils import data


import pandas as pd


import torchvision.transforms as standard_transforms


from torch.utils.data import DataLoader


import random


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from math import exp


from torch.nn.parameter import Parameter


from torch.nn import functional as F


from torch.nn.modules.loss import _Loss


import numbers


import math


from torch import nn


import torchvision.utils as vutils


from torchvision import models


from matplotlib import pyplot as plt


import matplotlib


import scipy.io as sio


from torch import optim


from torch.optim.lr_scheduler import StepLR


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):

    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class convDU(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(9, 1)):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)), nn.ReLU(inplace=True))

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n, c, 1, w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
        for i in range(h):
            pos = h - i - 1
            if pos == h - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        fea = torch.cat(fea_stack, 2)
        return fea


class convLR(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(1, 9)):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)), nn.ReLU(inplace=True))

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n, c, h, 1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
        for i in range(w):
            pos = w - i - 1
            if pos == w - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        fea = torch.cat(fea_stack, 3)
        return fea


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    kernel /= kernel.sum()
    return kernel


class SSIM_Loss(_Loss):

    def __init__(self, in_channels, size=11, sigma=1.5, size_average=True):
        super(SSIM_Loss, self).__init__(size_average)
        self.in_channels = in_channels
        self.size = int(size)
        self.sigma = sigma
        self.size_average = size_average
        kernel = gaussian_kernel(self.size, self.sigma)
        self.kernel_size = kernel.shape
        weight = np.tile(kernel, (in_channels, 1, 1, 1))
        self.weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(self, input, target, mask=None):
        _assert_no_grad(target)
        mean1 = F.conv2d(input, self.weight, padding=self.size, groups=self.in_channels)
        mean2 = F.conv2d(target, self.weight, padding=self.size, groups=self.in_channels)
        mean1_sq = mean1 * mean1
        mean2_sq = mean2 * mean2
        mean_12 = mean1 * mean2
        sigma1_sq = F.conv2d(input * input, self.weight, padding=self.size, groups=self.in_channels) - mean1_sq
        sigma2_sq = F.conv2d(target * target, self.weight, padding=self.size, groups=self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(input * target, self.weight, padding=self.size, groups=self.in_channels) - mean_12
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim = (2 * mean_12 + C1) * (2 * sigma_12 + C2) / ((mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Module):
        for mini_m in m.children():
            real_init_weights(mini_m)
    else:
        None


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


class CMTL(nn.Module):
    """
    Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density
    Estimation for Crowd Counting (Sindagi et al.)
    """

    def __init__(self, bn=False, num_classes=10):
        super(CMTL, self).__init__()
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(Conv2d(3, 16, 9, same_padding=True, NL='prelu', bn=bn), Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn))
        self.hl_prior_1 = nn.Sequential(Conv2d(32, 16, 9, same_padding=True, NL='prelu', bn=bn), nn.MaxPool2d(2), Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn), nn.MaxPool2d(2), Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn), Conv2d(16, 8, 7, same_padding=True, NL='prelu', bn=bn))
        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32, 32)), Conv2d(8, 4, 1, same_padding=True, NL='prelu', bn=bn))
        self.hl_prior_fc1 = FC(4 * 1024, 512, NL='prelu')
        self.hl_prior_fc2 = FC(512, 256, NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')
        self.de_stage_1 = nn.Sequential(Conv2d(32, 20, 7, same_padding=True, NL='prelu', bn=bn), nn.MaxPool2d(2), Conv2d(20, 40, 5, same_padding=True, NL='prelu', bn=bn), nn.MaxPool2d(2), Conv2d(40, 20, 5, same_padding=True, NL='prelu', bn=bn), Conv2d(20, 10, 5, same_padding=True, NL='prelu', bn=bn))
        self.de_stage_2 = nn.Sequential(Conv2d(18, 24, 3, same_padding=True, NL='prelu', bn=bn), Conv2d(24, 32, 3, same_padding=True, NL='prelu', bn=bn), nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True), nn.PReLU(), nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0, bias=True), nn.PReLU(), Conv2d(8, 1, 1, same_padding=True, NL='relu', bn=bn))
        initialize_weights(self.modules())

    def forward(self, im_data):
        x_base = self.base_layer(im_data)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1)
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)
        x_den = self.de_stage_1(x_base)
        x_den = torch.cat((x_hlp1, x_den), 1)
        x_den = self.de_stage_2(x_den)
        return x_den, x_cls


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class SAModule_Head(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn, kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn, kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn, kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn, kernel_size=7, padding=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn, kernel_size=1)
        self.branch3x3 = nn.Sequential(BasicConv(in_channels, 2 * branch_out, use_bn=use_bn, kernel_size=1), BasicConv(2 * branch_out, branch_out, use_bn=use_bn, kernel_size=3, padding=1))
        self.branch5x5 = nn.Sequential(BasicConv(in_channels, 2 * branch_out, use_bn=use_bn, kernel_size=1), BasicConv(2 * branch_out, branch_out, use_bn=use_bn, kernel_size=5, padding=2))
        self.branch7x7 = nn.Sequential(BasicConv(in_channels, 2 * branch_out, use_bn=use_bn, kernel_size=1), BasicConv(2 * branch_out, branch_out, use_bn=use_bn, kernel_size=7, padding=3))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SANet(nn.Module):

    def __init__(self, gray_input=False, use_bn=True):
        super(SANet, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3
        self.encoder = nn.Sequential(SAModule_Head(in_channels, 64, use_bn), nn.MaxPool2d(2, 2), SAModule(64, 128, use_bn), nn.MaxPool2d(2, 2), SAModule(128, 128, use_bn), nn.MaxPool2d(2, 2), SAModule(128, 128, use_bn))
        self.decoder = nn.Sequential(BasicConv(128, 64, use_bn=use_bn, kernel_size=9, padding=4), BasicDeconv(64, 64, 2, stride=2, use_bn=use_bn), BasicConv(64, 32, use_bn=use_bn, kernel_size=7, padding=3), BasicDeconv(32, 32, 2, stride=2, use_bn=use_bn), BasicConv(32, 16, use_bn=use_bn, kernel_size=5, padding=2), BasicDeconv(16, 16, 2, stride=2, use_bn=use_bn), BasicConv(16, 16, use_bn=use_bn, kernel_size=3, padding=1), BasicConv(16, 1, use_bn=False, kernel_size=1))
        initialize_weights(self.modules())

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class AlexNet(nn.Module):

    def __init__(self, pretrained=True):
        super(AlexNet, self).__init__()
        alex = models.alexnet(pretrained=pretrained)
        features = list(alex.features.children())
        self.layer1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=4)
        self.layer1plus = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Conv2d(64, 192, kernel_size=5, padding=3)
        self.layer2plus_to_5 = nn.Sequential(*features[4:12])
        self.de_pred = nn.Sequential(Conv2d(256, 128, 1, same_padding=True, NL='relu'), Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        self.layer1.load_state_dict(alex.features[0].state_dict())
        self.layer2.load_state_dict(alex.features[3].state_dict())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1plus(x)
        x = self.layer2(x)
        x = self.layer2plus_to_5(x)
        x = self.de_pred(x)
        x = F.upsample(x, scale_factor=16)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MCNN(nn.Module):
    """
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    """

    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(Conv2d(3, 16, 9, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(16, 32, 7, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(32, 16, 7, same_padding=True, bn=bn), Conv2d(16, 8, 7, same_padding=True, bn=bn))
        self.branch2 = nn.Sequential(Conv2d(3, 20, 7, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(20, 40, 5, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(40, 20, 5, same_padding=True, bn=bn), Conv2d(20, 10, 5, same_padding=True, bn=bn))
        self.branch3 = nn.Sequential(Conv2d(3, 24, 5, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(24, 48, 3, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(48, 24, 3, same_padding=True, bn=bn), Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))
        initialize_weights(self.modules())

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        x = F.upsample(x, scale_factor=4)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class Res101(nn.Module):

    def __init__(self, pretrained=True):
        super(Res101, self).__init__()
        self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'), Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.de_pred(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)


class Res101_SFCN(nn.Module):

    def __init__(self, pretrained=True):
        super(Res101_SFCN, self).__init__()
        self.seen = 0
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = []
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.convDU = convDU(in_out_channels=64, kernel_size=(1, 9))
        self.convLR = convLR(in_out_channels=64, kernel_size=(9, 1))
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())
        initialize_weights(self.modules())
        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.backend(x)
        x = self.convDU(x)
        x = self.convLR(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x


class Res50(nn.Module):

    def __init__(self, pretrained=True):
        super(Res50, self).__init__()
        self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'), Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        initialize_weights(self.modules())
        res = models.resnet50(pretrained=pretrained)
        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.de_pred(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)


class VGG(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])
        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'), Conv2d(128, 1, 1, same_padding=True, NL='relu'))

    def forward(self, x):
        x = self.features4(x)
        x = self.de_pred(x)
        x = F.upsample(x, scale_factor=8)
        return x


class VGG_decoder(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG_decoder, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])
        self.de_pred = nn.Sequential(Conv2d(512, 128, 3, same_padding=True, NL='relu'), nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=True), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True), nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True), nn.ReLU(), Conv2d(16, 1, 1, same_padding=True, NL='relu'))

    def forward(self, x):
        x = self.features4(x)
        x = self.de_pred(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BasicConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicDeconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CMTL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CSRNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Res101,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Res101_SFCN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Res50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SAModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'use_bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SAModule_Head,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'use_bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SANet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG_decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (convDU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     False),
    (convLR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     False),
]

class Test_gjy3035_C_3_Framework(_paritybench_base):
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

