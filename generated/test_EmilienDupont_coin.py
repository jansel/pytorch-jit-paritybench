import sys
_module = sys.modules[__name__]
del sys
main = _module
plots = _module
siren = _module
training = _module
util = _module

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


import torch


from torchvision import transforms


from torchvision.utils import save_image


from torch import nn


from math import sqrt


from collections import OrderedDict


import numpy as np


from torch._C import dtype


from typing import Dict


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        w_std = 1 / dim_in if self.is_first else sqrt(c / dim_in) / w0
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.0, w0_initial=30.0, use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(SirenLayer(dim_in=layer_dim_in, dim_out=dim_hidden, w0=layer_w0, use_bias=use_bias, is_first=is_first))
        self.net = nn.Sequential(*layers)
        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Sine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Siren,
     lambda: ([], {'dim_in': 4, 'dim_hidden': 4, 'dim_out': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SirenLayer,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_EmilienDupont_coin(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

