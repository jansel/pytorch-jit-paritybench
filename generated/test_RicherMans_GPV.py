import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
forward = _module
models = _module
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


from sklearn.metrics import confusion_matrix


from sklearn.metrics import roc_auc_score


from sklearn.metrics import precision_recall_fscore_support


import pandas as pd


import numpy as np


import uuid


import torch.nn as nn


import collections


import logging


import scipy


import sklearn.preprocessing as pre


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision ** 2).sum(self.pooldim) / time_decision.sum(self.pooldim)


class MeanPool(nn.Module):

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class Block2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=padding, bias=False), nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)


class CRNN(nn.Module):

    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(Block2D(1, 32), nn.LPPool2d(4, (2, 4)), Block2D(32, 128), Block2D(128, 128), nn.LPPool2d(4, (2, 4)), Block2D(128, 128), Block2D(128, 128), nn.LPPool2d(4, (1, 4)), nn.Dropout(0.3))
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500, inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(rnn_input_dim, 128, bidirectional=True, batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'linear'), inputdim=256, outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-07, 1.0)
        decision_time = torch.nn.functional.interpolate(decision_time.transpose(1, 2), time, mode='linear', align_corners=False).transpose(1, 2)
        decision = self.temp_pool(x, decision_time).clamp(1e-07, 1.0).squeeze(1)
        return decision, decision_time


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block2D,
     lambda: ([], {'cin': 4, 'cout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearSoftPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_RicherMans_GPV(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

