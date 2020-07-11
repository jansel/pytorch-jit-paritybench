import sys
_module = sys.modules[__name__]
del sys
datasets = _module
eval = _module
networks = _module
resnet = _module
ss_transforms = _module
train = _module
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


import torch.utils.data as data


import numpy as np


import logging


import torch.nn.functional as F


import torch.optim as optim


from torchvision import transforms


from sklearn.metrics import average_precision_score


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


import numbers


import random


from torch.autograd import Variable


from torch.optim.lr_scheduler import _LRScheduler


class ResNet50(nn.Module):

    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet.resnet50(pretrained=pretrained)
        self.relu = self.model.relu
        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)
        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)
        return feature_map, out


class Classifier(nn.Module):

    def __init__(self, in_features=2048, num_class=20):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        return x


class FPA(nn.Module):

    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels / 4)
        self.channels_cond = channels
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)
        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)
        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)
        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)
        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))
        out = self.relu(x_master + x_gpb)
        return out


class GAU(nn.Module):

    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out


class PAN(nn.Module):

    def __init__(self, blocks=[]):
        """
        :param blocks: Blocks of the network with reverse sequential.
        """
        super(PAN, self).__init__()
        channels_blocks = []
        for i, block in enumerate(blocks):
            channels_blocks.append(list(list(block.children())[2].children())[4].weight.shape[0])
        self.fpa = FPA(channels=channels_blocks[0])
        self.gau_block1 = GAU(channels_blocks[0], channels_blocks[1], upsample=False)
        self.gau_block2 = GAU(channels_blocks[1], channels_blocks[2])
        self.gau_block3 = GAU(channels_blocks[2], channels_blocks[3])
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms=[]):
        """
        :param fms: Feature maps of forward propagation in the network with reverse sequential. shape:[b, c, h, w]
        :return: fm_high. [b, 256, h, w]
        """
        for i, fm_low in enumerate(fms):
            if i == 0:
                fm_high = self.fpa(fm_low)
            else:
                fm_high = self.gau[int(i - 1)](fm_high, fm_low)
        return fm_high


class Mask_Classifier(nn.Module):

    def __init__(self, in_features=256, num_class=21):
        super(Mask_Classifier, self).__init__()
        self.mask_conv = nn.Conv2d(in_features, num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mask_conv(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=rate, dilation=rate, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FPA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (GAU,
     lambda: ([], {'channels_high': 4, 'channels_low': 4}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {}),
     False),
    (Mask_Classifier,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
]

class Test_JaveyWang_Pyramid_Attention_Networks_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

