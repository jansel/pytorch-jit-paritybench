import sys
_module = sys.modules[__name__]
del sys
conf = _module
settings = _module
FocalLoss = _module
LabelSmoothing = _module
criterion = _module
dataset = _module
lr_find = _module
FindLR = _module
WarmUpLR = _module
lr_scheduler = _module
vgg = _module
train = _module
transforms = _module
transforms = _module
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


import numpy as np


from torch.utils.data import Dataset


import torch.optim as optim


from torch.utils.data import DataLoader


import matplotlib


import matplotlib.pyplot as plt


from torch.optim.lr_scheduler import _LRScheduler


import random


import math


import numbers


from torch.autograd import Variable


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added
        one_hot = one_hot
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.

        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length
        return one_hot

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'.format(x.size(0), target.size(0)))
        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'.format(x.size(0)))
        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'.format(x.size()))
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(-x * smoothed_target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class BasicConv(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG(nn.Module):

    def __init__(self, blocks, num_class=200):
        super().__init__()
        self.input_channels = 3
        self.conv1 = self._make_layers(64, blocks[0])
        self.conv2 = self._make_layers(128, blocks[1])
        self.conv3 = self._make_layers(256, blocks[2])
        self.conv4 = self._make_layers(512, blocks[3])
        self.conv5 = self._make_layers(512, blocks[4])
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, num_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, output_channels, layer_num):
        layers = []
        while layer_num:
            layers.append(BasicConv(self.input_channels, output_channels, kernel_size=3, padding=1, bias=False))
            self.input_channels = output_channels
            layer_num -= 1
        layers.append(nn.MaxPool2d(2, stride=2))
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_weiaicunzai_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

