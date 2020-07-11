import sys
_module = sys.modules[__name__]
del sys
criteria = _module
dataloader = _module
dense_to_sparse = _module
kitti_dataloader = _module
nyu_dataloader = _module
transforms = _module
main = _module
metrics = _module
models = _module
utils = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


import numpy as np


import torch.utils.data as data


import math


import random


import numbers


import types


import collections


import warnings


import scipy.ndimage.interpolation as itpl


import scipy.misc as misc


import time


import torch.backends.cudnn as cudnn


import torch.optim


import torch.nn.functional as F


import torchvision.models


import matplotlib.pyplot as plt


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

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), 'inconsistent dimensions'
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class Unpool(nn.Module):

    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, (0), (0)] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class Decoder(nn.Module):
    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DeConv(Decoder):

    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, 'kernel_size out of range: {}'.format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, 'deconv parameters incorrect'
            module_name = 'deconv{}'.format(kernel_size)
            return nn.Sequential(collections.OrderedDict([(module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride, padding, output_padding, bias=False)), ('batchnorm', nn.BatchNorm2d(in_channels // 2)), ('relu', nn.ReLU(inplace=True))]))
        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // 2 ** 2)
        self.layer4 = convt(in_channels // 2 ** 3)


class UpConv(Decoder):

    def upconv_module(self, in_channels):
        upconv = nn.Sequential(collections.OrderedDict([('unpool', Unpool(in_channels)), ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)), ('batchnorm', nn.BatchNorm2d(in_channels // 2)), ('relu', nn.ReLU())]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels // 2)
        self.layer3 = self.upconv_module(in_channels // 4)
        self.layer4 = self.upconv_module(in_channels // 8)


class UpProj(Decoder):


    class UpProjModule(nn.Module):

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels // 2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)), ('batchnorm1', nn.BatchNorm2d(out_channels)), ('relu', nn.ReLU()), ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)), ('batchnorm2', nn.BatchNorm2d(out_channels))]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)), ('batchnorm', nn.BatchNorm2d(out_channels))]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels // 2)
        self.layer3 = self.UpProjModule(in_channels // 4)
        self.layer4 = self.UpProjModule(in_channels // 8)


def choose_decoder(decoder, in_channels):
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == 'upproj':
        return UpProj(in_channels)
    elif decoder == 'upconv':
        return UpConv(in_channels)
    else:
        assert False, 'invalid option for decoder: {}'.format(decoder)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ResNet(nn.Module):

    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        self.output_size = output_size
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        del pretrained_model
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)
        self.decoder = choose_decoder(decoder, num_channels // 2)
        self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeConv,
     lambda: ([], {'in_channels': 18, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
    (MaskedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unpool,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpConv,
     lambda: ([], {'in_channels': 18}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
    (UpProj,
     lambda: ([], {'in_channels': 18}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
]

class Test_fangchangma_sparse_to_dense_pytorch(_paritybench_base):
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

