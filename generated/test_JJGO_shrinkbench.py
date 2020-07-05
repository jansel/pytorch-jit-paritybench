import sys
_module = sys.modules[__name__]
del sys
python = _module
main = _module
pareto = _module
datasets = _module
places365 = _module
experiment = _module
base = _module
prune = _module
train = _module
metrics = _module
abstract_flops = _module
accuracy = _module
flops = _module
memory = _module
size = _module
models = _module
cifar_resnet = _module
cifar_vgg = _module
head = _module
mnistnet = _module
plot = _module
data = _module
pruning = _module
abstract = _module
mask = _module
mixin = _module
modules = _module
utils = _module
vision = _module
strategies = _module
channel = _module
magnitude = _module
random = _module
util = _module
automap = _module
color = _module
csvlogger = _module
online = _module

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


import numpy as np


import torch


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import math


import torchvision.models


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from torch.nn.modules.utils import _pair


from collections import defaultdict


import warnings


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.linear.is_classifier = True
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBNReLU, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=3 // 2)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGGBnDrop(nn.Module):

    def __init__(self, num_classes=10):
        super(VGGBnDrop, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(ConvBNReLU(3, 64), nn.Dropout(0.3), ConvBNReLU(64, 64), nn.MaxPool2d(2, 2, ceil_mode=True), ConvBNReLU(64, 128), nn.Dropout(0.4), ConvBNReLU(128, 128), nn.MaxPool2d(2, 2, ceil_mode=True), ConvBNReLU(128, 256), nn.Dropout(0.4), ConvBNReLU(256, 256), nn.Dropout(0.4), ConvBNReLU(256, 256), nn.MaxPool2d(2, 2, ceil_mode=True), ConvBNReLU(256, 512), nn.Dropout(0.4), ConvBNReLU(512, 512), nn.Dropout(0.4), ConvBNReLU(512, 512), nn.MaxPool2d(2, 2, ceil_mode=True), ConvBNReLU(512, 512), nn.Dropout(0.4), ConvBNReLU(512, 512), nn.Dropout(0.4), ConvBNReLU(512, 512), nn.MaxPool2d(2, 2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, self.num_classes))
        self.classifier[-1].is_classifier = True

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_weights(self):

        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                init.normal_(module.weight, 0, math.sqrt(2) / fan_in)
                init.zeros_(module.bias)
        self.apply(init_weights)


WEIGHTS_DIR = '../pretrained'


def weights_path(model, path=None):
    if path is None:
        path = WEIGHTS_DIR
        if 'WEIGHTSPATH' in os.environ:
            path = os.environ['WEIGHTSPATH'] + ':' + path
    paths = [pathlib.Path(p) for p in path.split(':')]
    for p in paths:
        for root, dirs, files in os.walk(p, followlinks=True):
            if model in files:
                wpath = pathlib.Path(root) / model
                print(f'Found {model} under {wpath}')
                return wpath
    else:
        raise LookupError(f'Could not find {model} in {paths}')


class MnistNet(nn.Module):
    """Small network designed for Mnist debugging
    """

    def __init__(self, pretrained=False):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc2.is_classifier = True
        if pretrained:
            weights = weights_path('mnistnet.pt')
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _ensure_tensor(x):
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def _same_device(x_mask, x):
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


def _same_shape(x_mask, x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x.shape == x_mask.shape


class MaskedModule(nn.Module):

    def __init__(self, layer, weight_mask, bias_mask=None):
        super(MaskedModule, self).__init__()
        self.weight = layer.weight
        self.bias = layer.bias
        self.register_buffer('weight_mask', None)
        self.register_buffer('bias_mask', None)
        self.set_masks(weight_mask, bias_mask)

    def forward_pre(self):
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias
        return weight, bias

    def set_masks(self, weight_mask, bias_mask=None):
        assert _same_shape(weight_mask, self.weight), f'Weight Mask must match dimensions'
        weight_mask = _ensure_tensor(weight_mask)
        self.weight_mask = _same_device(weight_mask, self.weight)
        self.weight.data.mul_(weight_mask)
        if bias_mask is not None:
            bias_mask = _ensure_tensor(bias_mask)
            assert self.bias is not None, 'Provided layer must have bias for it to be masked'
            assert _same_shape(bias_mask, self.bias), f'Bias Mask must match dimensions'
            self.bias_mask = _same_device(bias_mask, self.bias)
            self.bias.data.mul_(bias_mask)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGGBnDrop,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
]

class Test_JJGO_shrinkbench(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

