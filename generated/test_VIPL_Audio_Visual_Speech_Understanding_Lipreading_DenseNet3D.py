import sys
_module = sys.modules[__name__]
del sys
data = _module
dataset = _module
preprocess = _module
statefultransforms = _module
main = _module
Dense3D = _module
training = _module
validation = _module

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


from torch.utils.data import Dataset


import numpy as np


import random


from scipy import ndimage


import torchvision.transforms.functional as functional


import torchvision.transforms as transforms


import torch


import torch.nn as nn


from collections import OrderedDict


import scipy.io as sio


from collections import defaultdict


import matplotlib.pyplot as plt


import torch.nn.init as init


import torch.nn.functional as F


import math


from torch.autograd import Variable


import torch.optim as optim


from torch.utils.data import DataLoader


class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """

    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target.squeeze(0)).unsqueeze(1))
        loss = torch.cat(loss, 1)
        mask = torch.zeros(loss.size(0), loss.size(1)).float()
        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, L - 1] = 1.0
        loss = (loss * mask).sum() / mask.sum()
        return loss


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))


def _validate(modelOutput, length, labels, total=None, wrong=None):
    averageEnergies = torch.sum(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        averageEnergies[i] = modelOutput[i, :length[i]].sum(0)
    maxvalues, maxindices = torch.max(averageEnergies, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        elif wrong is not None:
            if l not in wrong:
                wrong[l] = 1
            else:
                wrong[l] += 1
    return averageEnergies, count


class Dense3D(torch.nn.Module):

    def __init__(self, options, growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0):
        super(Dense3D, self).__init__()
        block_config = 4, 8, 12, 8
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(3, num_init_features, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)), ('norm0', nn.BatchNorm3d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.gru1 = nn.GRU(536 * 3 * 3, 256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 1000))
        self.loss = NLLSequenceLoss

    def validator_function(self):
        return _validate

    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        f2 = self.features(x)
        f2 = f2.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, H, W = f2.size()
        f2 = f2.view(B, T, -1)
        f2, _ = self.gru1(f2)
        f2, _ = self.gru2(f2)
        f2 = self.fc(f2).log_softmax(-1)
        return f2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_VIPL_Audio_Visual_Speech_Understanding_Lipreading_DenseNet3D(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

