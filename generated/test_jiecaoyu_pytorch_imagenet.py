import sys
_module = sys.modules[__name__]
del sys
datasets = _module
folder = _module
transforms = _module
main = _module
model_list = _module
alexnet = _module
convert_key = _module

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


import torch.utils.data as data


import torch


import math


import random


import numpy as np


import numbers


import types


import collections


import time


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torch.utils.model_zoo as model_zoo


class LRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(inplace=True), LRN(local_size=5, alpha=0.0001, beta=0.75), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), nn.ReLU(inplace=True), LRN(local_size=5, alpha=0.0001, beta=0.75), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LRN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jiecaoyu_pytorch_imagenet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

