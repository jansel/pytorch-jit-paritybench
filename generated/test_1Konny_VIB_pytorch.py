import sys
_module = sys.modules[__name__]
del sys
datasets = _module
main = _module
model = _module
solver = _module
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


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import MNIST


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


from torch.autograd import Variable


import time


from numbers import Number


import math


import torch.optim as optim


from torch.optim import lr_scheduler


from torch import nn


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor
    else:
        return tensor


def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


class ToyNet(nn.Module):

    def __init__(self, K=256):
        super(ToyNet, self).__init__()
        self.K = K
        self.encode = nn.Sequential(nn.Linear(784, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True), nn.Linear(1024, 2 * self.K))
        self.decode = nn.Sequential(nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        statistics = self.encode(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)
        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)
        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):

        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())
        if n != 1:
            mu = expand(mu)
            std = expand(std)
        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

