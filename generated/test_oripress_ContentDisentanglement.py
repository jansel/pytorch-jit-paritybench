import sys
_module = sys.modules[__name__]
del sys
eval = _module
models = _module
preprocess = _module
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


import torch


import torch.nn as nn


from torch import nn


from torch import optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import torch.utils.data as data


import torchvision.utils as vutils


class E1(nn.Module):

    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size
        self.full = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512 - self.sep, 4, 2, 1), nn.InstanceNorm2d(512 - self.sep), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512 - self.sep, 512 - self.sep, 4, 2, 1), nn.InstanceNorm2d(512 - self.sep), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):

    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size
        self.full = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, self.sep, 4, 2, 1), nn.InstanceNorm2d(self.sep), nn.LeakyReLU(0.2))

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class Decoder(nn.Module):

    def __init__(self, size):
        super(Decoder, self).__init__()
        self.size = size
        self.main = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.ReLU(inplace=True), nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(inplace=True), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, net):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net


class Disc(nn.Module):

    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size
        self.classify = nn.Sequential(nn.Linear((512 - self.sep) * self.size * self.size, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, net):
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (Disc,
     lambda: ([], {'sep': 4, 'size': 4}),
     lambda: ([torch.rand([4, 8128])], {}),
     True),
]

class Test_oripress_ContentDisentanglement(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

