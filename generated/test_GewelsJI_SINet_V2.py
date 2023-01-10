import sys
_module = sys.modules[__name__]
del sys
MyTesting = _module
MyTrain_Val = _module
Network_Res2Net_GRA_NCD = _module
Res2Net_v1b = _module
lib = _module
data_val = _module
Network_Res2Net_GRA_NCD = _module
Res2Net_v1b = _module
FeatureViz = _module
MyFeatureVisulization = _module
cod10k_subclass_split = _module
data_val = _module
dataloader = _module
fps = _module
generate_LaTeX = _module
heatmap = _module
pytorch_jittor_convert = _module
tif2png = _module
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


import torch.nn.functional as F


import numpy as np


from scipy import misc


from torchvision.utils import make_grid


import logging


import torch.backends.cudnn as cudnn


import math


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import time


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)), BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3))
        self.branch2 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)), BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)), BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch3 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):

    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class GRA(nn.Module):

    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True))
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y, xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y, xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y, xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y, xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception('Invalid Channel')
        x = x + self.conv(x_cat)
        y = y + self.score(x)
        return x, y


class ReverseStage(nn.Module):

    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        y = -1 * torch.sigmoid(y) + 1
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)
        return y


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
            downsample = nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
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


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('/media/nercms/NERCMS/GepengJi/Medical_Seqmentation/CRANet/models/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
    return model


class Network(nn.Module):

    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')
        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRA,
     lambda: ([], {'channel': 4, 'subchannel': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeighborConnectionDecoder,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 32, 32])], {}),
     True),
    (RFB_modified,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_GewelsJI_SINet_V2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

