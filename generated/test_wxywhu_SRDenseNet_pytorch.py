import sys
_module = sys.modules[__name__]
del sys
SR_DenseNet = _module
SR_DenseNet_2 = _module
dataset = _module
test = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


from math import sqrt


import numpy as np


import torch.nn.init as init


import torch.utils.data as data


from torch.autograd import Variable


import time


import math


import scipy.io as sio


from scipy.ndimage import gaussian_filter


from math import exp


class SingleLayer(nn.Module):

    def __init__(self, inChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SingleBlock(nn.Module):

    def __init__(self, inChannels, growthRate, nDenselayer):
        super(SingleBlock, self).__init__()
        self.block = self._make_dense(inChannels, growthRate, nDenselayer)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


def xavier(param):
    init.xavier_uniform(param)


class Net(nn.Module):

    def __init__(self, inChannels, growthRate, nDenselayer, nBlock):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, growthRate, kernel_size=3, padding=1, bias=True)
        inChannels = growthRate
        self.denseblock = self._make_block(inChannels, growthRate, nDenselayer, nBlock)
        inChannels += growthRate * nDenselayer * nBlock
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1, padding=0, bias=True)
        self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_block(self, inChannels, growthRate, nDenselayer, nBlock):
        blocks = []
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels, growthRate, nDenselayer))
            inChannels += growthRate * nDenselayer
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.denseblock(out)
        out = self.Bottleneck(out)
        out = self.convt1(out)
        out = self.convt2(out)
        HR = self.conv2(out)
        return HR


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {'inChannels': 4, 'growthRate': 4, 'nDenselayer': 1, 'nBlock': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (SingleBlock,
     lambda: ([], {'inChannels': 4, 'growthRate': 4, 'nDenselayer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'inChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wxywhu_SRDenseNet_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

