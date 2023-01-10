import sys
_module = sys.modules[__name__]
del sys
config = _module
layer = _module
main = _module
metric = _module
model = _module
preprocess = _module

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


import numpy as np


import torch.optim as optim


import torch.utils.data as data


from sklearn.metrics import accuracy_score


from torch.utils import data


import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, num_vetex, act=F.relu, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()
        self.alpha = 1.0
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.bias = None
        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()
        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)
        return m_norm

    def forward(self, adj, x):
        x = self.dropout(x)
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = self.alpha * adj_norm + (1.0 - self.alpha) * sqr_norm
        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias
        x_out = self.act(x_out)
        return x_out


class StandConvolution(nn.Module):

    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=5, stride=2), nn.InstanceNorm2d(dims[1]), nn.ReLU(inplace=True), nn.Conv2d(dims[1], dims[2], kernel_size=5, stride=2), nn.InstanceNorm2d(dims[2]), nn.ReLU(inplace=True), nn.Conv2d(dims[2], dims[3], kernel_size=5, stride=2), nn.InstanceNorm2d(dims[3]), nn.ReLU(inplace=True))
        self.fc = nn.Linear(dims[3] * 3, num_classes)

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = self.fc(x_tmp.view(x.size(0), -1))
        return x_out


class StandRecurrent(nn.Module):

    def __init__(self, dims, num_classes, dropout):
        super(StandRecurrent, self).__init__()
        self.lstm = nn.LSTM(dims[0] * 45, dims[1], batch_first=True, dropout=0)
        self.fc = nn.Linear(dims[1], num_classes)

    def forward(self, x):
        x_tmp, _ = self.lstm(x.contiguous().view(x.size(0), x.size(1), -1))
        x_out = self.fc(x_tmp[:, -1])
        return x_out


class GGCN(nn.Module):

    def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, feat_dims, dropout=0.5):
        super(GGCN, self).__init__()
        terminal_cnt = 5
        actor_cnt = 1
        adj = adj + torch.eye(adj.size(0)).detach()
        ident = torch.eye(adj.size(0))
        zeros = torch.zeros(adj.size(0), adj.size(1))
        self.adj = torch.cat([torch.cat([adj, ident, zeros], 1), torch.cat([ident, adj, ident], 1), torch.cat([zeros, ident, adj], 1)], 0).float()
        self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))
        self.gcl = GraphConvolution(gc_dims[0] + feat_dims, gc_dims[1], num_v, dropout=dropout)
        self.conv = StandConvolution(sc_dims, num_classes, dropout=dropout)
        nn.init.xavier_normal_(self.terminal)

    def forward(self, x):
        head_la = F.interpolate(torch.stack([self.terminal[0], self.terminal[1]], 2), 6)
        head_ra = F.interpolate(torch.stack([self.terminal[0], self.terminal[2]], 2), 6)
        lw_ra = F.interpolate(torch.stack([self.terminal[3], self.terminal[4]], 2), 6)
        node_features = torch.cat([(head_la[:, :, :3] + head_ra[:, :, :3]) / 2, torch.stack((lw_ra[:, :, 2], lw_ra[:, :, 1], lw_ra[:, :, 0]), 2), lw_ra[:, :, 3:], head_la[:, :, 3:], head_ra[:, :, 3:]], 2)
        x = torch.cat((x, node_features.permute(0, 2, 1).unsqueeze(1).repeat(1, 32, 1, 1)), 3)
        concat_seq = torch.cat([x[:, :-2], x[:, 1:-1], x[:, 2:]], 2)
        multi_conv = self.gcl(self.adj, concat_seq)
        logit = self.conv(multi_conv)
        return logit


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (StandRecurrent,
     lambda: ([], {'dims': [4, 4], 'num_classes': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 180])], {}),
     True),
]

class Test_yongqyu_st_gcn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

