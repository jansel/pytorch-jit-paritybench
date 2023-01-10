import sys
_module = sys.modules[__name__]
del sys
dataloading = _module
defaults = _module
evaluation = _module
net = _module
novelty_detector = _module
partition_mnist = _module
save_to_csv = _module
schedule = _module
train_AAE = _module
utils = _module
jacobian = _module
multiprocessing = _module
save_plot = _module
threshold_search = _module
tracker = _module

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


import torch.utils.data


import numpy as np


import warnings


import torch


from torch import nn


from torch.nn import functional as F


from torchvision.utils import save_image


from torch.autograd import Variable


import logging


import scipy.optimize


import matplotlib.pyplot as plt


import scipy.stats


from scipy.special import loggamma


from torch import optim


import time


import torch.nn as nn


import torch.nn.functional as F


import functools


from collections import OrderedDict


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):

    def __init__(self, z_size, d=128, channels=1):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(z_size, d * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 2, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x)) * 0.5 + 0.5
        return x


class Discriminator(nn.Module):

    def __init__(self, d=128, channels=1):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


class Encoder(nn.Module):

    def __init__(self, z_size, d=128, channels=1):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, z_size, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x


class ZDiscriminator(nn.Module):

    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


class ZDiscriminator_mergebatch(nn.Module):

    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator_mergebatch, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d * batchSize, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2).view(1, -1)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Encoder,
     lambda: ([], {'z_size': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Generator,
     lambda: ([], {'z_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZDiscriminator,
     lambda: ([], {'z_size': 4, 'batchSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZDiscriminator_mergebatch,
     lambda: ([], {'z_size': 4, 'batchSize': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_podgorskiy_GPND(_paritybench_base):
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

