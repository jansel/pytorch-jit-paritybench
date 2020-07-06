import sys
_module = sys.modules[__name__]
del sys
nalu = _module
core = _module
nac_cell = _module
nalu_cell = _module
layers = _module
nalu_layer = _module
setup = _module
tests = _module

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


from torch import Tensor


from torch import nn


from torch.nn.parameter import Parameter


from torch.nn.init import xavier_uniform_


from torch.nn.functional import linear


from torch import sigmoid


from torch import tanh


from torch import exp


from torch import log


from torch.nn import Sequential


class NacCell(nn.Module):
    """Basic NAC unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf
    """

    def __init__(self, in_shape, out_shape):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.W_ = Parameter(Tensor(out_shape, in_shape))
        self.M_ = Parameter(Tensor(out_shape, in_shape))
        xavier_uniform_(self.W_), xavier_uniform_(self.M_)
        self.register_parameter('bias', None)

    def forward(self, input):
        W = tanh(self.W_) * sigmoid(self.M_)
        return linear(input, W, self.bias)


class NaluCell(nn.Module):
    """Basic NALU unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf
    """

    def __init__(self, in_shape, out_shape):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.G = Parameter(Tensor(out_shape, in_shape))
        self.nac = NacCell(out_shape, in_shape)
        xavier_uniform_(self.G)
        self.eps = 1e-05
        self.register_parameter('bias', None)

    def forward(self, input):
        a = self.nac(input)
        g = sigmoid(linear(input, self.G, self.bias))
        ag = g * a
        log_in = log(abs(input) + self.eps)
        m = exp(self.nac(log_in))
        md = (1 - g) * m
        return ag + md


class NaluLayer(nn.Module):

    def __init__(self, input_shape, output_shape, n_layers, hidden_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.hidden_shape = hidden_shape
        layers = [NaluCell(hidden_shape if n > 0 else input_shape, hidden_shape if n < n_layers - 1 else output_shape) for n in range(n_layers)]
        self.model = Sequential(*layers)

    def forward(self, data):
        return self.model(data)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NacCell,
     lambda: ([], {'in_shape': 4, 'out_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NaluCell,
     lambda: ([], {'in_shape': 4, 'out_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NaluLayer,
     lambda: ([], {'input_shape': 4, 'output_shape': 4, 'n_layers': 1, 'hidden_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bharathgs_NALU(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

