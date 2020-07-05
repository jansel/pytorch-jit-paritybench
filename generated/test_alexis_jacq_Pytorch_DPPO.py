import sys
_module = sys.modules[__name__]
del sys
chief = _module
main = _module
model = _module
ppo = _module
test = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.optim as optim


import torch.multiprocessing as mp


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.autograd import Variable


import random


import math


import time


from collections import deque


class Model(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        self.p_fc1 = nn.Linear(num_inputs, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)
        self.v_fc1 = nn.Linear(num_inputs, h_size_1 * 5)
        self.v_fc2 = nn.Linear(h_size_1 * 5, h_size_2)
        self.mu = nn.Linear(h_size_2, num_outputs)
        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.v = nn.Linear(h_size_2, 1)
        for name, p in self.named_parameters():
            if 'bias' in name:
                p.data.fill_(0)
            """
            if 'mu.weight' in name:
                p.data.normal_()
                p.data /= torch.sum(p.data**2,0).expand_as(p.data)"""
        self.train()

    def forward(self, inputs):
        x = F.tanh(self.p_fc1(inputs))
        x = F.tanh(self.p_fc2(x))
        mu = self.mu(x)
        sigma_sq = torch.exp(self.log_std)
        x = F.tanh(self.v_fc1(inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Model,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_alexis_jacq_Pytorch_DPPO(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

