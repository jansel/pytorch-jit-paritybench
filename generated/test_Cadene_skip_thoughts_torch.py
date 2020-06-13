import sys
_module = sys.modules[__name__]
del sys
pytorch = _module
setup = _module
skipthoughts = _module
dropout = _module
gru = _module
skipthoughts = _module
version = _module
test = _module
dump_features = _module
dump_grus = _module
dump_hashmaps = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


from torch import nn


from torch.autograd import Variable


from itertools import repeat


import torch.nn as nn


import torch.nn.functional as F


import numpy


from collections import OrderedDict


class SequentialDropout(nn.Module):

    def __init__(self, p=0.5):
        super(SequentialDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        self.restart = True

    def _make_noise(self, input):
        return Variable(input.data.new().resize_as_(input.data))

    def forward(self, input):
        if self.p > 0 and self.training:
            if self.restart:
                self.noise = self._make_noise(input)
                self.noise.data.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.data.fill_(0)
                self.noise = self.noise.expand_as(input)
                self.restart = False
            return input.mul(self.noise)
        return input

    def end_of_sequence(self):
        self.restart = True

    def backward(self, grad_output):
        self.end_of_sequence()
        if self.p > 0 and self.training:
            return grad_output.mul(self.noise)
        else:
            return grad_output

    def __repr__(self):
        return type(self).__name__ + '({:.4f})'.format(self.p)


class AbstractGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError


class AbstractGRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(AbstractGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        raise NotImplementedError

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, (i), :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx


urls = {}


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Cadene_skip_thoughts_torch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SequentialDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

