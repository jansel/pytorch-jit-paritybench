import sys
_module = sys.modules[__name__]
del sys
agnn = _module
model = _module
train = _module
utils = _module

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


from torch.nn.parameter import Parameter


from torch.autograd import Variable


import torch.nn.functional as F


import time


import numpy as np


import torch.optim as optim


import scipy.sparse as sp


class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        norm2 = torch.norm(x, 2, 1).view(-1, 1)
        cos = self.beta * torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-07)
        mask = (1.0 - adj) * -1000000000.0
        masked = cos + mask
        P = F.softmax(masked, dim=1)
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, initializer=nn.init.xavier_uniform):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(initializer(torch.Tensor(in_features, out_features)))

    def forward(self, input):
        return torch.mm(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class AGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout_rate):
        super(AGNN, self).__init__()
        self.layers = nlayers
        self.dropout_rate = dropout_rate
        self.embeddinglayer = LinearLayer(nfeat, nhid)
        nn.init.xavier_uniform(self.embeddinglayer.weight)
        self.attentionlayers = nn.ModuleList()
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=False))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())
        self.outputlayer = LinearLayer(nhid, nclass)
        nn.init.xavier_uniform(self.outputlayer.weight)

    def forward(self, x, adj):
        x = F.relu(self.embeddinglayer(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj)
        x = self.outputlayer(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AGNN,
     lambda: ([], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'nlayers': 1, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GraphAttentionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (LinearLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_dawnranger_pytorch_AGNN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

