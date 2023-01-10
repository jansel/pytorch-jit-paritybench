import sys
_module = sys.modules[__name__]
del sys
catboost_ = _module
datasets = _module
ensemble = _module
evaluate = _module
synthetic = _module
train0 = _module
train1 = _module
train1_synthetic = _module
train3 = _module
train4 = _module
tune = _module
xgboost_ = _module
lib = _module
data = _module
deep = _module
env = _module
metrics = _module
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


import math


from copy import deepcopy


from typing import Any


from typing import Literal


from typing import Optional


from typing import Union


import torch


import torch.nn as nn


import numpy as np


from sklearn.tree import DecisionTreeClassifier


from sklearn.tree import DecisionTreeRegressor


import time


from torch import Tensor


from torch.nn import Parameter


from typing import List


from typing import Tuple


from typing import cast


import uuid


import warnings


from collections import Counter


import pandas as pd


import sklearn.preprocessing


from sklearn.impute import SimpleImputer


from sklearn.preprocessing import StandardScaler


from typing import Callable


import torch.nn.functional as F


import torch.optim as optim


import enum


from typing import TypeVar


from typing import get_args


from typing import get_origin


class NLinear(nn.Module):

    def __init__(self, n: int, d_in: int, d_out: int, bias: bool=True) ->None:
        super().__init__()
        self.weight = Parameter(Tensor(n, d_in, d_out))
        self.bias = Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NLayerNorm(nn.Module):

    def __init__(self, n_features: int, d: int) ->None:
        super().__init__()
        self.weight = Parameter(torch.ones(n_features, d))
        self.bias = Parameter(torch.zeros(n_features, d))

    def forward(self, x: Tensor) ->Tensor:
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


class NLinearMemoryEfficient(nn.Module):

    def __init__(self, n: int, d_in: int, d_out: int) ->None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class _DICE(nn.Module):
    """The DICE method from "Methods for Numeracy-Preserving Word Embeddings" by Sundararaman et al."""
    Q: Tensor

    def __init__(self, d: int, x_min: float, x_max: float) ->None:
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        M = torch.randn(d, d)
        Q, _ = torch.linalg.qr(M)
        self.register_buffer('Q', Q)

    def forward(self, x: Tensor) ->Tensor:
        assert x.ndim == 1
        d = len(self.Q)
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.where((0.0 <= x) & (x <= 1.0), x * torch.pi, torch.empty_like(x).uniform_(-torch.pi, torch.pi))
        exponents = torch.arange(d - 1, dtype=x.dtype, device=x.device)
        x = torch.column_stack([torch.cos(x)[:, None] * torch.sin(x)[:, None] ** exponents[None], torch.sin(x) ** (d - 1)])
        x = x @ self.Q
        return x


class DICEEmbeddings(nn.Module):

    def __init__(self, d: int, lower_bounds: list[float], upper_bounds: list[float]) ->None:
        super().__init__()
        self.modules_ = nn.ModuleList([_DICE(d, *bounds) for bounds in zip(lower_bounds, upper_bounds)])
        self.d_embedding = d

    def forward(self, x: Tensor) ->Tensor:
        assert x.shape[1] == len(self.modules_)
        return torch.stack([self.modules_[i](x[:, i]) for i in range(len(self.modules_))], 1)


def cos_sin(x: Tensor) ->Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NLayerNorm,
     lambda: ([], {'n_features': 4, 'd': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (NLinear,
     lambda: ([], {'n': 4, 'd_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (NLinearMemoryEfficient,
     lambda: ([], {'n': 4, 'd_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Yura52_tabular_dl_num_embeddings(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

