import sys
_module = sys.modules[__name__]
del sys
data = _module
main = _module
models = _module
nin = _module
util = _module
datasets = _module
folder = _module
transforms = _module
main = _module
model_list = _module
alexnet = _module
util = _module
main = _module
LeNet_5 = _module
util = _module

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


import numpy


import torchvision.transforms as transforms


import torch.nn as nn


import torch.optim as optim


from torch.autograd import Variable


import torch.nn.functional as F


import torch.utils.data as data


import math


import random


import numpy as np


import numbers


import types


import collections


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torch.utils.model_zoo as model_zoo


from torchvision import datasets


from torchvision import transforms


class BinActive(torch.autograd.Function):
    """
    Binarize the input activations and calculate the mean across channel dimension.
    """

    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0, Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=0.0001, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            if self.previous_conv:
                self.bn = nn.BatchNorm2d(int(input_channels / size), eps=0.0001, momentum=0.1, affine=True)
            else:
                self.bn = nn.BatchNorm1d(input_channels, eps=0.0001, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(192, eps=0.0001, momentum=0.1, affine=False), nn.ReLU(inplace=True), BinConv2d(192, 160, kernel_size=1, stride=1, padding=0), BinConv2d(160, 96, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5), BinConv2d(192, 192, kernel_size=1, stride=1, padding=0), BinConv2d(192, 192, kernel_size=1, stride=1, padding=0), nn.AvgPool2d(kernel_size=3, stride=2, padding=1), BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5), BinConv2d(192, 192, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(192, eps=0.0001, momentum=0.1, affine=False), nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=8, stride=1, padding=0))

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.BatchNorm2d(96, eps=0.0001, momentum=0.1, affine=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1), nn.MaxPool2d(kernel_size=3, stride=2), BinConv2d(256, 384, kernel_size=3, stride=1, padding=1), BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1), BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.classifier = nn.Sequential(BinConv2d(256 * 6 * 6, 4096, Linear=True), BinConv2d(4096, 4096, dropout=0.5, Linear=True), nn.BatchNorm1d(4096, eps=0.001, momentum=0.1, affine=True), nn.Dropout(), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class LeNet_5(nn.Module):

    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(20, eps=0.0001, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_conv2 = BinConv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_ip1 = BinConv2d(50 * 4 * 4, 500, Linear=True, previous_conv=True, size=4 * 4)
        self.ip2 = nn.Linear(500, 10)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.bin_conv2(x)
        x = self.pool2(x)
        x = self.bin_ip1(x)
        x = self.ip2(x)
        return x

