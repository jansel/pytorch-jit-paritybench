import sys
_module = sys.modules[__name__]
del sys
bad_grad_viz = _module
checkpoints = _module
config = _module
dataloader = _module
datasets = _module
filelist = _module
folderlist = _module
loaders = _module
transforms = _module
evaluate = _module
classification = _module
losses = _module
classification = _module
regression = _module
main = _module
model = _module
models = _module
naivecnn = _module
naiveresnet = _module
net = _module
resnet = _module
plugins = _module
image = _module
logger = _module
monitor = _module
visualizer = _module
train = _module
utils = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


import torchvision


import torch.optim as optim


class Classification(nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.loss(input, target)
        return loss


class Regression(nn.Module):

    def __init__(self):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        loss = self.loss.forward(input, target)
        return loss


class NoiseLayer(nn.Module):

    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = torch.randn(1, in_planes, 1, 1)
        self.level = level
        self.layers = nn.Sequential(nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1), nn.BatchNorm2d(out_planes))

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape
        if tmp1[1] != tmp2[1] or tmp1[2] != tmp2[2] or tmp1[3] != tmp2[3]:
            self.noise = (2 * torch.rand(x.data.shape) - 1) * self.level
            self.noise = self.noise
        x.data = x.data + self.noise
        x = self.layers(x)
        return x


class NoiseModel(nn.Module):

    def __init__(self, nblocks, nlayers, nchannels, nfilters, nclasses, level):
        super(NoiseModel, self).__init__()
        self.num = nfilters
        self.level = level
        layers = []
        layers.append(NoiseLayer(3, nfilters, self.level))
        for i in range(1, nlayers):
            layers.append(self._make_layer(nfilters, nfilters, nblocks, self.level))
            layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.num, nclasses)

    def _make_layer(self, in_planes, out_planes, nblocks, level):
        layers = []
        for i in range(nblocks):
            layers.append(NoiseLayer(in_planes, out_planes, level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num)
        x = self.classifier(x)
        return x


class NoiseLayer(nn.Module):

    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False)
        self.level = level
        self.layers = nn.Sequential(nn.ReLU(True), nn.BatchNorm2d(in_planes), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1))

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level
        y = torch.add(x, self.noise)
        z = self.layers(y)
        return z


class NoiseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(NoiseLayer(in_planes, planes, level), nn.MaxPool2d(stride, stride), nn.BatchNorm2d(planes), nn.ReLU(True), NoiseLayer(planes, planes, level), nn.BatchNorm2d(planes))
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class NoiseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBottleneck, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(True), NoiseLayer(planes, planes, level), nn.MaxPool2d(stride, stride), nn.BatchNorm2d(planes), nn.ReLU(True), nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False), nn.BatchNorm2d(planes * 4))
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class NoiseResNet(nn.Module):

    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(nn.Conv2d(nchannels, nfilters, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(nfilters), nn.ReLU(True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 1 * nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2 * nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4 * nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8 * nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8 * nfilters * block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

    def __init__(self, block, layers, nchannels, nfilters, nclasses=1000):
        self.inplanes = nfilters
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, nfilters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nfilters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nfilters, layers[0])
        self.layer2 = self._make_layer(block, 2 * nfilters, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nfilters, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nfilters, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8 * nfilters * block.expansion, nclasses)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)
        x6 = self.fc(x)
        return [x0, x1, x2, x3, x4, x5]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (Regression,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_juefeix_pnn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

