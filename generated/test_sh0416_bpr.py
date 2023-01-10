import sys
_module = sys.modules[__name__]
del sys
master = _module
metrics = _module
preprocess = _module
setup = _module
tests = _module
test_vsl = _module
train = _module
setup = _module
setup = _module

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


from torch.utils import cpp_extension


import random


from collections import deque


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import IterableDataset


from torch.utils.data import DataLoader


from torch.utils.data import get_worker_info


from torch.utils.tensorboard import SummaryWriter


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


class BPR(nn.Module):

    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = weight_decay

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def recommend(self, u):
        """Return recommended item list given users.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BPR,
     lambda: ([], {'user_size': 4, 'item_size': 4, 'dim': 4, 'weight_decay': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
]

class Test_sh0416_bpr(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

