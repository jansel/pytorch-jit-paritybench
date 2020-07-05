import sys
_module = sys.modules[__name__]
del sys
Config = _module
ConvKB = _module
ConvKB_1D = _module
Model = _module
train_ConvKB = _module
batching = _module
builddata = _module
eval = _module
model = _module
train = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


import time


import numpy as np


import torch.autograd as autograd


import torch.nn.functional as F


from numpy.random import RandomState


class MyDataParallel(nn.DataParallel):

    def _getattr__(self, name):
        return getattr(self.module, name)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.batch_h = None
        self.batch_t = None
        self.batch_r = None
        self.batch_y = None
    """
	def get_positive_instance(self):
		self.positive_h = self.batch_h[0:self.config.batch_size]
		self.positive_t = self.batch_t[0:self.config.batch_size]
		self.positive_r = self.batch_r[0:self.config.batch_size]
		return self.positive_h, self.positive_t, self.positive_r

	def get_negative_instance(self):
		self.negative_h = self.batch_h[self.config.batch_size, self.config.batch_seq_size]
		self.negative_t = self.batch_t[self.config.batch_size, self.config.batch_seq_size]
		self.negative_r = self.batch_r[self.config.batch_size, self.config.batch_seq_size]
		return self.negative_h, self.negative_t, self.negative_r
 	"""

    def get_positive_score(self, score):
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        negative_score = score[self.config.batch_size:self.config.batch_seq_size]
        negative_score = negative_score.view(-1, self.config.batch_size)
        negative_score = torch.mean(negative_score, 0)
        return negative_score

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MyDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_daiquocnguyen_ConvKB(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

