import sys
_module = sys.modules[__name__]
del sys
functional = _module
functions = _module
psamask = _module
modules = _module
psamask = _module
src = _module
psanet = _module
pspnet = _module
resnet = _module
demo = _module
test = _module
train = _module
config = _module
dataset = _module
transform = _module
util = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch import nn


import torch


import torch.nn.functional as F


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


import logging


import numpy as np


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.utils.data


import time


import random


import torch.optim


import torch.multiprocessing as mp


import torch.distributed as dist


import torch.nn.init as initer


class PSAMask(nn.Module):

    def __init__(self, psa_type=0, mask_H_=None, mask_W_=None):
        super(PSAMask, self).__init__()
        assert psa_type in [0, 1]
        assert mask_H_ in None and mask_W_ is None or mask_H_ is not None and mask_W_ is not None
        self.psa_type = psa_type
        self.mask_H_ = mask_H_
        self.mask_W_ = mask_W_

    def forward(self, input):
        return F.psa_mask(input, self.psa_type, self.mask_H_, self.mask_W_)


class PSA(nn.Module):

    def __init__(self, in_channels=2048, mid_channels=512, psa_type=2,
        compact=False, shrink_factor=2, mask_h=59, mask_w=59,
        normalization_factor=1.0, psa_softmax=True):
        super(PSA, self).__init__()
        assert psa_type in [0, 1, 2]
        self.psa_type = psa_type
        self.compact = compact
        self.shrink_factor = shrink_factor
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.psa_softmax = psa_softmax
        if normalization_factor is None:
            normalization_factor = mask_h * mask_w
        self.normalization_factor = normalization_factor
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, mid_channels,
            kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels), nn.
            ReLU(inplace=True))
        self.attention = nn.Sequential(nn.Conv2d(mid_channels, mid_channels,
            kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels), nn.
            ReLU(inplace=True), nn.Conv2d(mid_channels, mask_h * mask_w,
            kernel_size=1, bias=False))
        if psa_type == 2:
            self.reduce_p = nn.Sequential(nn.Conv2d(in_channels,
                mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(
                mid_channels), nn.ReLU(inplace=True))
            self.attention_p = nn.Sequential(nn.Conv2d(mid_channels,
                mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(
                mid_channels), nn.ReLU(inplace=True), nn.Conv2d(
                mid_channels, mask_h * mask_w, kernel_size=1, bias=False))
        self.proj = nn.Sequential(nn.Conv2d(mid_channels * (2 if psa_type ==
            2 else 1), in_channels, kernel_size=1, bias=False), nn.
            BatchNorm2d(in_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        out = x
        if self.psa_type in [0, 1]:
            x = self.reduce(x)
            n, c, h, w = x.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x = F.interpolate(x, size=(h, w), mode='bilinear',
                    align_corners=True)
            y = self.attention(x)
            if self.compact:
                if self.psa_type == 1:
                    y = y.view(n, h * w, h * w).transpose(1, 2).view(n, h *
                        w, h, w)
            else:
                y = PF.psa_mask(y, self.psa_type, self.mask_h, self.mask_w)
            if self.psa_softmax:
                y = F.softmax(y, dim=1)
            x = torch.bmm(x.view(n, c, h * w), y.view(n, h * w, h * w)).view(n,
                c, h, w) * (1.0 / self.normalization_factor)
        elif self.psa_type == 2:
            x_col = self.reduce(x)
            x_dis = self.reduce_p(x)
            n, c, h, w = x_col.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x_col = F.interpolate(x_col, size=(h, w), mode='bilinear',
                    align_corners=True)
                x_dis = F.interpolate(x_dis, size=(h, w), mode='bilinear',
                    align_corners=True)
            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            if self.compact:
                y_dis = y_dis.view(n, h * w, h * w).transpose(1, 2).view(n,
                    h * w, h, w)
            else:
                y_col = PF.psa_mask(y_col, 0, self.mask_h, self.mask_w)
                y_dis = PF.psa_mask(y_dis, 1, self.mask_h, self.mask_w)
            if self.psa_softmax:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(x_col.view(n, c, h * w), y_col.view(n, h * w,
                h * w)).view(n, c, h, w) * (1.0 / self.normalization_factor)
            x_dis = torch.bmm(x_dis.view(n, c, h * w), y_dis.view(n, h * w,
                h * w)).view(n, c, h, w) * (1.0 / self.normalization_factor)
            x = torch.cat([x_col, x_dis], 1)
        x = self.proj(x)
        if self.shrink_factor != 1:
            h = (h - 1) * self.shrink_factor + 1
            w = (w - 1) * self.shrink_factor + 1
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                align_corners=True)
        return torch.cat((out, x), 1)


class PSANet(nn.Module):

    def __init__(self, layers=50, dropout=0.1, classes=2, zoom_factor=8,
        use_psa=True, psa_type=2, compact=False, shrink_factor=2, mask_h=59,
        mask_w=59, normalization_factor=1.0, psa_softmax=True, criterion=nn
        .CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSANet, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        assert psa_type in [0, 1, 2]
        self.zoom_factor = zoom_factor
        self.use_psa = use_psa
        self.criterion = criterion
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3,
            resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (resnet.layer1,
            resnet.layer2, resnet.layer3, resnet.layer4)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        fea_dim = 2048
        if use_psa:
            self.psa = PSA(fea_dim, 512, psa_type, compact, shrink_factor,
                mask_h, mask_w, normalization_factor, psa_softmax)
            fea_dim *= 2
        self.cls = nn.Sequential(nn.Conv2d(fea_dim, 512, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=
            True), nn.Dropout2d(p=dropout), nn.Conv2d(512, classes,
            kernel_size=1))
        if self.training:
            self.aux = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
                inplace=True), nn.Dropout2d(p=dropout), nn.Conv2d(256,
                classes, kernel_size=1))

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_psa:
            x = self.psa(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                align_corners=True)
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear',
                    align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


class PPM(nn.Module):

    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear',
                align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):

    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2,
        zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(
        ignore_index=255), pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3,
            resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (resnet.layer1,
            resnet.layer2, resnet.layer3, resnet.layer4)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(nn.Conv2d(fea_dim, 512, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=
            True), nn.Dropout2d(p=dropout), nn.Conv2d(512, classes,
            kernel_size=1))
        if self.training:
            self.aux = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
                inplace=True), nn.Dropout2d(p=dropout), nn.Conv2d(256,
                classes, kernel_size=1))

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                align_corners=True)
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear',
                    align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hszhao_semseg(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(PPM(*[], **{'in_dim': 4, 'reduction_dim': 4, 'bins': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

