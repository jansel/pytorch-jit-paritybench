import sys
_module = sys.modules[__name__]
del sys
datasets = _module
model = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import time


import numpy as np


from torchtext.vocab import GloVe


RNNS = ['LSTM', 'GRU']


class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.0, bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type)
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.transpose(0, 1).transpose(1, 2)
        energy = torch.bmm(query, keys)
        energy = F.softmax(energy.mul_(self.scale), dim=2)
        values = values.transpose(0, 1)
        linear_combination = torch.bmm(energy, values).squeeze(1)
        return energy, linear_combination


class Classifier(nn.Module):

    def __init__(self, embedding, encoder, attention, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)
        size = 0
        for p in self.parameters():
            size += p.nelement()
        None

    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))
        if isinstance(hidden, tuple):
            hidden = hidden[1]
        if self.encoder.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        energy, linear_combination = self.attention(hidden, outputs, outputs)
        logits = self.decoder(linear_combination)
        return logits, energy


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Encoder,
     lambda: ([], {'embedding_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_mttk_rnn_classifier(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

