import sys
_module = sys.modules[__name__]
del sys
config = _module
gcn = _module
mlp = _module
build_graph = _module
remove_words = _module
train = _module
tsne = _module
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


import torch


import torch.nn as nn


from sklearn import metrics


import random


import time


import numpy as np


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, support, act_func=None, featureless=False, dropout_rate=0.0, bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))
        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)
        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out
        return out


class GCN(nn.Module):

    def __init__(self, input_dim, support, dropout_rate=0.0, num_classes=10):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.0, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_iworldtong_text_gcn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

