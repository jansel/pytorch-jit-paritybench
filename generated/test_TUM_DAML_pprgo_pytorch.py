import sys
_module = sys.modules[__name__]
del sys
pprgo = _module
dataset = _module
ppr = _module
pprgo = _module
predict = _module
pytorch_utils = _module
sparsegraph = _module
train = _module
utils = _module
run_seml = _module
setup = _module

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


import time


import numpy as np


import math


import scipy.sparse as sp


import logging


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


class SparseDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        value_dropped = F.dropout(input.storage.value(), self.p, self.training)
        return torch_sparse.SparseTensor(row=input.storage.row(), rowptr=input.storage.rowptr(), col=input.storage.col(), value=value_dropped, sparse_sizes=input.sparse_sizes(), is_sorted=True)


class MixedDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            res = input.matmul(self.weight)
            if self.bias:
                res += self.bias[None, :]
        elif self.bias:
            res = torch.addmm(self.bias, input, self.weight)
        else:
            res = input.matmul(self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class PPRGoMLP(nn.Module):

    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()
        fcs = [MixedLinear(num_features, hidden_size, bias=False)]
        for i in range(nlayers - 2):
            fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
        fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.drop = MixedDropout(dropout)

    def forward(self, X):
        embs = self.drop(X)
        embs = self.fcs[0](embs)
        for fc in self.fcs[1:]:
            embs = fc(self.drop(F.relu(embs)))
        return embs


class PPRGo(nn.Module):

    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes, hidden_size, nlayers, dropout)

    def forward(self, X, ppr_scores, ppr_idx):
        logits = self.mlp(X)
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None], dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        return propagated_logits

