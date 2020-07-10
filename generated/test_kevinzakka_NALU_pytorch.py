import sys
_module = sys.modules[__name__]
del sys
failures = _module
function_learning = _module
models = _module
mlp = _module
nac = _module
nalu = _module
utils = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import math


import random


import torch.nn as nn


import torch.nn.init as init


from torch.nn.parameter import Parameter


def str2act(s):
    if s is 'none':
        return None
    elif s is 'hardtanh':
        return nn.Hardtanh()
    elif s is 'sigmoid':
        return nn.Sigmoid()
    elif s is 'relu6':
        return nn.ReLU6()
    elif s is 'tanh':
        return nn.Tanh()
    elif s is 'tanhshrink':
        return nn.Tanhshrink()
    elif s is 'hardshrink':
        return nn.Hardshrink()
    elif s is 'leakyrelu':
        return nn.LeakyReLU()
    elif s is 'softshrink':
        return nn.Softshrink()
    elif s is 'softsign':
        return nn.Softsign()
    elif s is 'relu':
        return nn.ReLU()
    elif s is 'prelu':
        return nn.PReLU()
    elif s is 'softplus':
        return nn.Softplus()
    elif s is 'elu':
        return nn.ELU()
    elif s is 'selu':
        return nn.SELU()
    else:
        raise ValueError('[!] Invalid activation function.')


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP).

    Also known as a Fully-Connected Network (FCN). This
    implementation assumes that all hidden layers have
    the same hidden size and the same activation function.

    Attributes:
        num_layers: the number of layers in the network.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
        activation: the activation function.
    """

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = str2act(activation)
        nonlin = True
        if self.activation is None:
            nonlin = False
        layers = []
        for i in range(num_layers - 1):
            layers.extend(self._layer(hidden_dim if i > 0 else in_dim, hidden_dim, nonlin))
        layers.extend(self._layer(hidden_dim, out_dim, False))
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [nn.Linear(in_dim, out_dim), self.activation]
        else:
            return [nn.Linear(in_dim, out_dim)]

    def forward(self, x):
        out = self.model(x)
        return out


class NeuralAccumulatorCell(nn.Module):
    """A Neural Accumulator (NAC) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)
        self._reset_params()

    def _reset_params(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)


class NAC(nn.Module):
    """A stack of NAC layers.

    Attributes:
        num_layers: the number of NAC layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        layers = []
        for i in range(num_layers):
            layers.append(NeuralAccumulatorCell(hidden_dim if i > 0 else in_dim, hidden_dim if i < num_layers - 1 else out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class NeuralArithmeticLogicUnitCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10
        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)
        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input):
        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)


class NALU(nn.Module):
    """A stack of NAC layers.

    Attributes:
        num_layers: the number of NAC layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        layers = []
        for i in range(num_layers):
            layers.append(NeuralArithmeticLogicUnitCell(hidden_dim if i > 0 else in_dim, hidden_dim if i < num_layers - 1 else out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'num_layers': 1, 'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NAC,
     lambda: ([], {'num_layers': 1, 'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NALU,
     lambda: ([], {'num_layers': 1, 'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeuralAccumulatorCell,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeuralArithmeticLogicUnitCell,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kevinzakka_NALU_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

