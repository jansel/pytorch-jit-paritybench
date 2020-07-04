import sys
_module = sys.modules[__name__]
del sys
basenet = _module
vgg16_bn = _module
craft = _module
craft_utils = _module
file_utils = _module
imgproc = _module
refinenet = _module
test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from collections import namedtuple


import torch


import torch.nn as nn


import torch.nn.init as init


from torchvision import models


from torchvision.models.vgg import model_urls


import torch.nn.functional as F


from torch.autograd import Variable


import time


import torch.backends.cudnn as cudnn


import numpy as np


from collections import OrderedDict


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(torch.nn.Module):

    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://',
            'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained
            ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slice5 = torch.nn.Sequential(nn.MaxPool2d(kernel_size=3,
            stride=1, padding=1), nn.Conv2d(512, 1024, kernel_size=3,
            padding=6, dilation=6), nn.Conv2d(1024, 1024, kernel_size=1))
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple('VggOutputs', ['fc7', 'relu5_3', 'relu4_3',
            'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


class double_conv(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch + mid_ch, mid_ch,
            kernel_size=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1), nn.
            BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):

    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)
        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)
        num_class = 2
        self.conv_cls = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32,
            16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d
            (16, 16, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(16,
            num_class, kernel_size=1))
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)
        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear',
            align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear',
            align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear',
            align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature


class RefineNet(nn.Module):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.last_conv = nn.Sequential(nn.Conv2d(34, 64, kernel_size=3,
            padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.
            Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, padding
            =1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.aspp1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,
            dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=
            True), nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Conv2d(128, 1, kernel_size=1))
        self.aspp2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,
            dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=
            True), nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Conv2d(128, 1, kernel_size=1))
        self.aspp3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,
            dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=
            True), nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Conv2d(128, 1, kernel_size=1))
        self.aspp4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,
            dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=
            True), nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Conv2d(128, 1, kernel_size=1))
        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = torch.cat([y.permute(0, 3, 1, 2), upconv4], dim=1)
        refine = self.last_conv(refine)
        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_clovaai_CRAFT_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(CRAFT(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(double_conv(*[], **{'in_ch': 4, 'mid_ch': 4, 'out_ch': 4}), [torch.rand([4, 8, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(vgg16_bn(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

