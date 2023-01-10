import sys
_module = sys.modules[__name__]
del sys
data_handler = _module
dataset = _module
dataset_factory = _module
incremental_loader = _module
experiment = _module
mnist_missing_experiment = _module
model = _module
misc_functions = _module
model_factory = _module
res_utils = _module
resnet32 = _module
test_model = _module
plotter = _module
run_experiment = _module
trainer = _module
evaluator = _module
trainer = _module
Colorer = _module
utils = _module
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


from torchvision import datasets


from torchvision import transforms


import torch


import numpy


import copy


import logging


import numpy as np


import torch.utils.data as td


from torch.autograd import Variable


from torchvision import models


import torch.nn as nn


import math


import torch.nn.functional as F


from torch.nn import init


import matplotlib.pyplot as plt


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()
        self.featureSize = 64
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.num_classes = num_classes
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc2 = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, scale=None, keep=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if feature:
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        x = self.fc(x) / T
        if keep is not None:
            x = x[:, keep[0]:keep[1]]
        if labels:
            return F.softmax(x, dim=1)
        if scale is not None:
            temp = F.softmax(x, dim=1)
            temp = temp * scale
            return temp
        return F.log_softmax(x, dim=1)

    def forwardFeature(self, x):
        pass


class Net(nn.Module):

    def __init__(self, noClasses, channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(4, 6, kernel_size=5, padding=(2, 2))
        self.conv2_bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=5, padding=(2, 2))
        self.conv2_bn2 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 10, kernel_size=5, padding=(2, 2))
        self.conv2_bn3 = nn.BatchNorm2d(10)
        self.conv5 = nn.Conv2d(10, 12, kernel_size=5, padding=(2, 2))
        self.conv5_bn3 = nn.BatchNorm2d(12)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(48, 100)
        self.fc = nn.Linear(100, noClasses)
        self.featureSize = 48

    def forward(self, x, feature=False, T=1, labels=False, scale=None, predictClass=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn3(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5_bn3(self.conv5(x))), 2))
        x = x.view(x.size(0), -1)
        if feature:
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        if labels:
            if predictClass:
                return F.softmax(self.fc(x) / T), F.softmax(self.fc2(x) / T)
            return F.softmax(self.fc(x) / T)
        if scale is not None:
            x = self.fc(x)
            temp = F.softmax(x / T)
            temp = temp * scale
            return temp
        if predictClass:
            return F.log_softmax(self.fc(x) / T), F.log_softmax(self.fc2(x) / T)
        return F.log_softmax(self.fc(x) / T)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DownsampleA,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownsampleC,
     lambda: ([], {'nIn': 1, 'nOut': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (DownsampleD,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_khurramjaved96_incremental_learning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

