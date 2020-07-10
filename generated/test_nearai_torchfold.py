import sys
_module = sys.modules[__name__]
del sys
setup = _module
torchfold = _module
torchfold = _module
torchfold_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import torch


from torch.autograd import Variable


import torch.nn as nn


from torch import optim


from torchtext import data


from torchtext import datasets


import collections


class TreeLSTM(nn.Module):

    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.left = nn.Linear(num_units, 5 * num_units)
        self.right = nn.Linear(num_units, 5 * num_units)

    def forward(self, left_in, right_in):
        lstm_in = self.left(left_in[0])
        lstm_in += self.right(right_in[0])
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] + f2.sigmoid() * right_in[1]
        h = o.sigmoid() * c.tanh()
        return h, c


class SPINN(nn.Module):

    def __init__(self, n_classes, size, n_words):
        super(SPINN, self).__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.embeddings = nn.Embedding(n_words, size)
        self.out = nn.Linear(size, n_classes)

    def leaf(self, word_id):
        return self.embeddings(word_id), Variable(torch.FloatTensor(word_id.size()[0], self.size))

    def children(self, left_h, left_c, right_h, right_c):
        return self.tree_lstm((left_h, left_c), (right_h, right_c))

    def logits(self, encoding):
        return self.out(encoding)


class TestEncoder(nn.Module):

    def __init__(self):
        super(TestEncoder, self).__init__()
        self.embed = nn.Embedding(10, 10)
        self.out = nn.Linear(20, 10)

    def concat(self, *nodes):
        return torch.cat(nodes, 0)

    def value(self, idx):
        return self.embed(idx)

    def value2(self, idx):
        return self.embed(idx), self.embed(idx)

    def attr(self, left, right):
        return self.out(torch.cat([left, right], 1))

    def logits(self, enc, embed):
        return torch.mm(enc, embed.t())


class RNNEncoder(nn.Module):

    def __init__(self, num_units, input_size):
        super(RNNEncoder, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.encoder = nn.GRUCell(self.input_size, self.num_units)

    def encode(self, input_, state):
        return self.encoder(input_, state)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TreeLSTM,
     lambda: ([], {'num_units': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_nearai_torchfold(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

