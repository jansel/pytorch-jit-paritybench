import sys
_module = sys.modules[__name__]
del sys
A3C = _module
master = _module
environment = _module
my_optim = _module
train = _module

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


import scipy.signal


import torch


import numpy as np


import random


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.autograd import Variable


import torch.optim as optim


import torch.multiprocessing as mp


import math


import logging


import time


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3CLSTMNet(nn.Module):

    def __init__(self, state_shape, action_dim):
        super(A3CLSTMNet, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(self.state_shape[0], 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(3 * 3 * 32, 256, 1)
        self.linear_policy_1 = nn.Linear(256, self.action_dim)
        self.softmax_policy = nn.Softmax()
        self.linear_value_1 = nn.Linear(256, 1)
        self.apply(weights_init)
        self.linear_policy_1.weight.data = normalized_columns_initializer(self.linear_policy_1.weight.data, 0.01)
        self.linear_policy_1.bias.data.fill_(0)
        self.linear_value_1.weight.data = normalized_columns_initializer(self.linear_value_1.weight.data, 1.0)
        self.linear_value_1.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, hidden):
        x = x.view(-1, self.state_shape[0], self.state_shape[1], self.state_shape[2])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 3 * 3 * 32)
        x, c = self.lstm(x, (hidden[0], hidden[1]))
        pl = self.linear_policy_1(x)
        pl = self.softmax_policy(pl)
        v = self.linear_value_1(x)
        return pl, v, (x, c)

