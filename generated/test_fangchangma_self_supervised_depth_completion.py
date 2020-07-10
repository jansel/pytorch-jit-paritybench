import sys
_module = sys.modules[__name__]
del sys
criteria = _module
kitti_loader = _module
pose_estimator = _module
transforms = _module
helper = _module
inverse_warp = _module
main = _module
metrics = _module
model = _module
vis_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import numpy as np


from numpy import linalg as LA


from random import choice


import torch.utils.data as data


import math


import random


import numbers


import types


import collections


import warnings


import scipy.ndimage.interpolation as itpl


import time


import torch.nn.functional as F


import torch.nn.parallel


import torch.optim


import torch.utils.data


from torchvision.models import resnet


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), 'inconsistent dimensions'
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), 'inconsistent dimensions'
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class PhotometricLoss(nn.Module):

    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):
        assert recon.dim() == 4, 'expected recon dimension to be 4, but instead got {}.'.format(recon.dim())
        assert target.dim() == 4, 'expected target dimension to be 4, but instead got {}.'.format(target.dim())
        assert recon.size() == target.size(), 'expected recon and target to have the same size, but got {} and {} instead'.format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = torch.sum(diff, 1)
        valid_mask = (torch.sum(recon, 1) > 0).float() * (torch.sum(target, 1) > 0).float()
        if mask is not None:
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                None
                self.loss = 0
        else:
            None
            self.loss = 0
        return self.loss


class SmoothnessLoss(nn.Module):

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):

        def second_derivative(x):
            assert x.dim() == 4, 'expected 4-dimensional data, but instead got {}'.format(x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()
        self.loss = second_derivative(depth)
        return self.loss


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)
    for m in layers.modules():
        init_weights(m)
    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)
    for m in layers.modules():
        init_weights(m)
    return layers


class DepthCompletionNet(nn.Module):

    def __init__(self, args):
        assert args.layers in [18, 34, 50, 101, 152], 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input
        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=256 + 128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=128 + 64, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x):
        if 'd' in self.modality:
            conv1_d = self.conv1_d(x['d'])
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])
        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if self.modality == 'd' else conv1_img
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)
        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)
        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)
        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)
        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)
        y = self.convtf(y)
        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(100 * y - min_distance) + min_distance


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MaskedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PhotometricLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SmoothnessLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_fangchangma_self_supervised_depth_completion(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

