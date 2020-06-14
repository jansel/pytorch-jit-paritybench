import sys
_module = sys.modules[__name__]
del sys
data = _module
main = _module
network = _module
test = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import math


import random


import numpy as np


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.optim as optim


from torch.autograd import Variable


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d
            (16), nn.ReLU(True), nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d
            (32), nn.ReLU(True), nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d
            (64), nn.ReLU(True), nn.Conv2d(64, 128, 4, 2, 1), nn.
            BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 512, 4, 2, 1
            ), nn.MaxPool2d((2, 2)), nn.ConvTranspose2d(512, 256, 4, 1, 0,
            bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(
            128), nn.ReLU(True), nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=
            False), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(
            64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False), nn.BatchNorm2d
            (16), nn.ReLU(True), nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=
            False), nn.Tanh())

    def forward(self, input):
        output = self.main(input)
        return output


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.LeakyReLU(
            0.2, inplace=True), nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(
            32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 64, 4, 2, 1
            ), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256
            ), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, 4, 2, 1
            ), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(512, 1, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_scaleway_frontalization(_paritybench_base):
    pass
