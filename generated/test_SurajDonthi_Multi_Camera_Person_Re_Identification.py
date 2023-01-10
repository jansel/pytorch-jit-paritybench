import sys
_module = sys.modules[__name__]
del sys
mtmct_reid = _module
data = _module
engine = _module
eval = _module
metrics = _module
model = _module
re_ranking = _module
rough = _module
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


from typing import Optional


from typing import Union


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torch.utils.data.dataset import Subset


from torchvision import transforms


import torch


import torch.nn.functional as F


from torch.optim import lr_scheduler


from torch.optim.sgd import SGD


from torchvision.transforms import transforms


import math


import numpy as np


import torch.nn as nn


from torch.nn import init


from torchvision.models import resnet50


import matplotlib.pyplot as plt


def weights_init_classifier(layer):
    if type(layer) == nn.Linear:
        init.normal_(layer.weight.data, std=0.001)
        init.constant_(layer.bias.data, 0.0)


def weights_init_kaiming(layer):
    if type(layer) in [nn.Conv1d, nn.Conv2d]:
        init.kaiming_normal_(layer.weight.data, mode='fan_in')
    elif type(layer) == nn.Linear:
        init.kaiming_normal_(layer.weight.data, mode='fan_out')
        init.constant_(layer.bias.data, 0.0)
    elif type(layer) == nn.BatchNorm1d:
        init.normal_(layer.weight.data, mean=1.0, std=0.02)
        init.constant_(layer.bias.data, 0.0)


class ClassifierBlock(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, dropout: bool=True, activation: str=None, num_bottleneck=512):
        super().__init__()
        self._layers(input_dim, num_classes, dropout, activation, num_bottleneck)

    def _layers(self, input_dim, num_classes, dropout, activation, num_bottleneck):
        block = [nn.Linear(input_dim, num_bottleneck), nn.BatchNorm1d(num_bottleneck)]
        if activation == 'relu':
            block += [nn.ReLU]
        elif activation == 'lrelu':
            block += [nn.LeakyReLU(0.1)]
        if dropout:
            block += [nn.Dropout(p=0.5)]
        block = nn.Sequential(*block)
        block.apply(weights_init_kaiming)
        classifier = nn.Linear(num_bottleneck, num_classes)
        classifier.apply(weights_init_classifier)
        self.block = block
        self.classifier = classifier

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_parts = 6
        self._layers(num_classes)

    def _layers(self, num_classes):
        self.model = resnet50(pretrained=True)
        del self.model.fc
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1
        self.model.avgpool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.model = nn.Sequential(*list(self.model.children()))
        for i in range(self.num_parts):
            name = 'classifier' + str(i)
            setattr(self, name, ClassifierBlock(2048, num_classes, True, 'lrelu', 256))

    def forward(self, x, training=False):
        x = self.model(x)
        x = torch.squeeze(x)
        if training:
            x = self.dropout(x)
            part = []
            strips_out = []
            for i in range(self.num_parts):
                part.append(x[:, :, i])
                name = 'classifier' + str(i)
                classifier = getattr(self, name)
                part_out = classifier(part[i])
                strips_out.append(part_out)
            return strips_out
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassifierBlock,
     lambda: ([], {'input_dim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PCB,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_SurajDonthi_Multi_Camera_Person_Re_Identification(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

