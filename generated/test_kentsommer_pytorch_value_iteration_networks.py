import sys
_module = sys.modules[__name__]
del sys
dataset = _module
dataset = _module
make_training_data = _module
domains = _module
gridworld = _module
generators = _module
obstacle_gen = _module
model = _module
test = _module
train = _module
utility = _module
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


import numpy as np


import torch


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.nn.parameter import Parameter


import matplotlib.pyplot as plt


from torch.autograd import Variable


import time


import torchvision.transforms as transforms


class VIN(nn.Module):

    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        self.h = nn.Conv2d(in_channels=config.l_i, out_channels=config.l_h, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.r = nn.Conv2d(in_channels=config.l_h, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.q = nn.Conv2d(in_channels=1, out_channels=config.l_q, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
        self.w = Parameter(torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, X, S1, S2, config):
        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(torch.cat([r, v], 1), torch.cat([self.q.weight, self.w], 1), stride=1, padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)
        q = F.conv2d(torch.cat([r, v], 1), torch.cat([self.q.weight, self.w], 1), stride=1, padding=1)
        slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)
        slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)
        logits = self.fc(q_out)
        return logits, self.sm(logits)

