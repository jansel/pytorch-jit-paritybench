import sys
_module = sys.modules[__name__]
del sys
cost = _module
datasets = _module
data_io = _module
dataset = _module
MSNet2D = _module
MSNet3D = _module
models = _module
submodule = _module
prediction = _module
train = _module
KittiColormap = _module
utils = _module
experiment = _module
metrics = _module
visualization = _module

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


import torch.nn.parallel


import torch.utils.data


import random


import numpy as np


from torch.utils.data import Dataset


import math


import torch.nn as nn


import torch.nn.functional as F


import torch


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import time


import torch.optim as optim


import torchvision.utils as vutils


import copy


from torch.autograd import Variable


from torch import Tensor


class MobileV2_Residual(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass2D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels))
        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=False))
    else:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()
        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True))
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio))
        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        feature_volume = torch.cat((l2, l3, l4), dim=1)
        return feature_volume


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:, ::2, :, :] = refimg_fea
    interwoven_features[:, 1::2, :, :] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


class MSNet2D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet2D, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 1
        self.volume_size = 48
        self.hg_size = 48
        self.dres_expanse_ratio = 3
        self.feature_extraction = feature_extraction(add_relus=True)
        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(256, 128, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(128, 64, 1, 1, 0, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 1, 1, 0, 1))
        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU(), nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(32), nn.ReLU(), nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU())
        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass2D(self.hg_size)
        self.encoder_decoder2 = hourglass2D(self.hg_size)
        self.encoder_decoder3 = hourglass2D(self.hg_size)
        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)
        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x
        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = torch.unsqueeze(cost0, 1)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


class MobileV2_Residual_3D(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio):
        super(MobileV2_Residual_3D, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass3D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass3D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual_3D(in_channels, in_channels * 2, 2, self.expanse_ratio)
        self.conv2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)
        self.conv3 = MobileV2_Residual_3D(in_channels * 2, in_channels * 4, 2, self.expanse_ratio)
        self.conv4 = MobileV2_Residual_3D(in_channels * 4, in_channels * 4, 1, self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels))
        self.redir1 = MobileV2_Residual_3D(in_channels, in_channels, 1, self.expanse_ratio)
        self.redir2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False), nn.BatchNorm3d(out_channels))


class MSNet3D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet3D, self).__init__()
        self.maxdisp = maxdisp
        self.hourglass_size = 32
        self.dres_expanse_ratio = 3
        self.num_groups = 40
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(MobileV2_Residual_3D(self.num_groups, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.dres1 = nn.Sequential(MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass3D(self.hourglass_size)
        self.encoder_decoder2 = hourglass3D(self.hourglass_size)
        self.encoder_decoder3 = hourglass3D(self.hourglass_size)
        self.classif0 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_left = self.feature_extraction(L)
        features_right = self.feature_extraction(R)
        volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4, self.num_groups)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MSNet3D,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (MobileV2_Residual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileV2_Residual_3D,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (hourglass2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hourglass3D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_cogsys_tuebingen_mobilestereonet(_paritybench_base):
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

