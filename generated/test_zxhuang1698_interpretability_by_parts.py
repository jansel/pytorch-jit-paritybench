import sys
_module = sys.modules[__name__]
del sys
celeba = _module
eval_acc = _module
eval_interp = _module
model = _module
train = _module
visualize = _module
grouping = _module
loss = _module
utils = _module
cub200 = _module
eval_acc = _module
eval_interp = _module
model = _module
train = _module
visualize = _module

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


from torchvision import transforms


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


import time


import numpy as np


from numpy.linalg import norm


from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import StandardScaler


import torch.optim as optim


import torch.nn.functional as F


import random


import torch.backends.cudnn as cudnn


import torchvision.models as models


from torch.utils.tensorboard.writer import SummaryWriter


from matplotlib import pyplot as plt


from matplotlib.patches import Rectangle


from matplotlib.gridspec import GridSpec


import matplotlib.image as mpimg


import torch.utils.data


import torchvision.datasets as datasets


from collections import OrderedDict


import math


from scipy import stats


from torch.autograd import Function


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
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


class GroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-05)
        else:
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(batch_size, self.num_parts, self.in_channels)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        x = x.permute(0, 2, 1)
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)
        c = grouping_centers
        sum_ass = torch.sum(assign, dim=2, keepdim=True)
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-05)
        sigma = (beta / 2).sqrt()
        out = (qx / sum_ass - c) / sigma.unsqueeze(0).unsqueeze(2)
        assign = assign.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        outputs = nn.functional.normalize(out, dim=2)
        outputs_t = outputs.permute(0, 2, 1)
        return outputs_t, assign

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.num_parts) + ')'


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_parts=32):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.n_parts = num_parts
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.grouping = GroupingUnit(256 * block.expansion, num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)
        self.post_block = nn.Sequential(Bottleneck1x1(1024, 512, stride=1, downsample=nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(2048))), Bottleneck1x1(2048, 512, stride=1), Bottleneck1x1(2048, 512, stride=1), Bottleneck1x1(2048, 512, stride=1))
        self.attconv = nn.Sequential(Bottleneck1x1(1024, 256, stride=1), Bottleneck1x1(1024, 256, stride=1), nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.ReLU())
        self.groupingbn = nn.BatchNorm2d(2048)
        self.mylinear = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck) or isinstance(m, Bottleneck1x1):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        region_feature, assign = self.grouping(x)
        region_feature = region_feature.contiguous().unsqueeze(3)
        att = self.attconv(region_feature)
        att = F.softmax(att, dim=2)
        region_feature = self.post_block(region_feature)
        out = region_feature * att
        out = out.contiguous().squeeze(3)
        out = F.avg_pool1d(out, self.n_parts) * self.n_parts
        out = out.contiguous().unsqueeze(3)
        out = self.groupingbn(out)
        out = out.contiguous().view(out.size(0), -1)
        out = self.mylinear(out)
        return out, att, assign


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupingUnit,
     lambda: ([], {'in_channels': 4, 'num_parts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zxhuang1698_interpretability_by_parts(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

