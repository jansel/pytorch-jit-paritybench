import sys
_module = sys.modules[__name__]
del sys
metrics = _module
archs = _module
mnist_train = _module
archs = _module
dataset = _module
omniglot_train = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Parameter


import math


import numpy as np


from torch import nn


from torch.nn import functional as F


from torchvision import models


import torchvision


from collections import OrderedDict


import torch.backends.cudnn as cudnn


import torch.optim as optim


from torch.optim import lr_scheduler


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import Dataset


import random


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder


class AdaCos(nn.Module):

    def __init__(self, num_features, num_classes, m=0.5):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-07, 1.0 - 1e-07))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        return output


class ArcFace(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.5):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-07, 1.0 - 1e-07))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


class SphereFace(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-07, 1.0 - 1e-07))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


class CosFace(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.relu(x)
        return output


class MNISTNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 2))
        self.features = nn.Sequential(*[VGGBlock(1, 16, 16), self.pool, VGGBlock(16, 32, 32), self.pool, VGGBlock(32, 64, 64), self.pool])
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(3 * 3 * 64, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.features(input)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)
        return output


class ResNet_IR(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            last_channels = 512
        elif args.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            last_channels = 512
        elif args.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            last_channels = 2048
        elif args.backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            last_channels = 2048
        elif args.backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
            last_channels = 2048
        self.features = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4)
        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(8 * 8 * last_channels, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaCos,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ArcFace,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CosFace,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SphereFace,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGBlock,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_4uiiurz1_pytorch_adacos(_paritybench_base):
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

