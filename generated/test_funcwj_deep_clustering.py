import sys
_module = sys.modules[__name__]
del sys
compute_cmvn = _module
dataset = _module
dcnet = _module
separate = _module
train_dcnet = _module
trainer = _module
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


import random


import logging


import numpy as np


import torch as th


from torch.nn.utils.rnn import pack_sequence


from torch.nn.utils.rnn import pad_sequence


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


import sklearn


import scipy.io as sio


import time


import warnings


def l2_normalize(x, dim=0, eps=1e-12):
    assert dim < x.dim()
    norm = th.norm(x, 2, dim, keepdim=True)
    return x / (norm + eps)


class DCNet(th.nn.Module):

    def __init__(self, num_bins, rnn='lstm', embedding_dim=20, num_layers=2, hidden_size=600, dropout=0.0, non_linear='tanh', bidirectional=True):
        super(DCNet, self).__init__()
        if non_linear not in ['tanh', 'sigmoid']:
            raise ValueError('Unsupported non-linear type: {}'.format(non_linear))
        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError('Unsupported rnn type: {}'.format(rnn))
        self.rnn = getattr(th.nn, rnn)(num_bins, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.drops = th.nn.Dropout(p=dropout)
        self.embed = th.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_bins * embedding_dim)
        self.non_linear = {'tanh': th.nn.functional.tanh, 'sigmoid': th.nn.functional.sigmoid}[non_linear]
        self.embedding_dim = embedding_dim

    def forward(self, x, train=True):
        is_packed = isinstance(x, PackedSequence)
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        N = x.size(0)
        x = self.drops(x)
        x = self.embed(x)
        x = self.non_linear(x)
        if train:
            x = x.view(N, -1, self.embedding_dim)
        else:
            x = x.view(-1, self.embedding_dim)
        x = l2_normalize(x, -1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DCNet,
     lambda: ([], {'num_bins': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_funcwj_deep_clustering(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

