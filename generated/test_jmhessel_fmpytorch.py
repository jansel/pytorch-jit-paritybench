import sys
_module = sys.modules[__name__]
del sys
fmpytorch = _module
regression = _module
toy = _module
second_order = _module
fm = _module
second_order_fast = _module
second_order_naive = _module
setup = _module
test = _module
test_second_order = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from itertools import count


import torch


import torch.autograd


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import time


import torch.optim as optim


from torch import nn


HIDDEN_SIZE = 100


INPUT_SIZE = 50


class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.dropout = torch.nn.Dropout(0.5)
        self.fm = FactorizationMachine(HIDDEN_SIZE, 5)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.fm(x)
        return x


class FactorizationMachine(torch.nn.Module):
    """Second order factorization machine layer"""

    def __init__(self, input_features, factors):
        """
        - input_features (int): the length of the input vector.
        - factors (int): the dimension of the interaction terms.
        """
        super(FactorizationMachine, self).__init__()
        if not FAST_VERSION:
            None
        self.input_features, self.factors = input_features, factors
        self.linear = nn.Linear(self.input_features, 1)
        self.second_order = SecondOrderInteraction(self.input_features,
            self.factors)

    def forward(self, x):
        self.linear.cpu()
        self.second_order.cpu()
        back_to_gpu = False
        if x.is_cuda:
            x = x.cpu()
            back_to_gpu = True
        linear = self.linear(x)
        interaction = self.second_order(x)
        res = linear + interaction
        if back_to_gpu:
            res = res
            x = x
        return res


class SecondOrderFunction(torch.autograd.Function):

    def forward(self, x, v):
        return fast_forward(self, x, v)

    def backward(self, grad_output):
        return fast_backward(self, grad_output)


class SecondOrderInteraction(torch.nn.Module):

    def __init__(self, n_feats, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.n_feats = n_feats
        self.n_factors = n_factors
        self.v = nn.Parameter(torch.Tensor(self.n_feats, self.n_factors),
            requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        return SecondOrderFunction()(x, self.v)


class SecondOrderInteraction(torch.nn.Module):

    def __init__(self, n_feats, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.n_feats = n_feats
        self.n_factors = n_factors
        self.v = nn.Parameter(torch.Tensor(self.n_feats, self.n_factors))
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        self.batch_size = x.size()[0]
        self.n_feats = x.size()[-1]
        self.n_factors = self.v.size()[-1]
        output = Variable(x.data.new(self.batch_size, self.n_feats, self.
            n_feats).zero_())
        all_interactions = torch.mm(self.v, self.v.t())
        for b in range(self.batch_size):
            for i in range(self.n_feats):
                for j in range(i + 1, self.n_feats):
                    output[b, i, j] = all_interactions[i, j] * x[b, i] * x[b, j
                        ]
        res = output.sum(1).sum(1, keepdim=True)
        return res


N_FACTORS = 5


class ModelSlow(torch.nn.Module):

    def __init__(self):
        super(ModelSlow, self).__init__()
        self.second_order = SOISlow(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x


class ModelFast(torch.nn.Module):

    def __init__(self):
        super(ModelFast, self).__init__()
        self.second_order = SOIFast(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x


class ModelSlow(torch.nn.Module):

    def __init__(self):
        super(ModelSlow, self).__init__()
        self.second_order = SOISlow(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x


class ModelFast(torch.nn.Module):

    def __init__(self):
        super(ModelFast, self).__init__()
        self.second_order = SOIFast(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jmhessel_fmpytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SecondOrderInteraction(*[], **{'n_feats': 4, 'n_factors': 4}), [torch.rand([4, 4])], {})

