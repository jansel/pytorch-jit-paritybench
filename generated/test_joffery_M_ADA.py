import sys
_module = sys.modules[__name__]
del sys
main_Digits = _module
ada_conv = _module
digits_process_dataset = _module
download_and_process_mnist = _module
ops = _module

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


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.nn as nn


from torch.autograd import Variable


import numpy


import torch


import torch.nn.functional as F


import numpy as np


import time


import scipy


from scipy import misc


from scipy import io


import math


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x, return_feat=False):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        if return_feat:
            return out4, self.fc3(out4)
        else:
            return self.fc3(out4)


class WAE(nn.Module):

    def __init__(self):
        super(WAE, self).__init__()
        self.fc1 = nn.Linear(3072, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 3072)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, 3072))
        return self.decode(z), z


class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=20):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(True), nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (WAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3072])], {}),
     True),
]

class Test_joffery_M_ADA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

