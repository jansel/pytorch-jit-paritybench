import sys
_module = sys.modules[__name__]
del sys
common = _module
camera = _module
data_utils = _module
generators = _module
graph_utils = _module
h36m_dataset = _module
log = _module
loss = _module
mocap_dataset = _module
quaternion = _module
skeleton = _module
utils = _module
visualization = _module
prepare_data_2d_h36m_sh = _module
prepare_data_h36m = _module
main_gcn = _module
main_linear = _module
models = _module
graph_non_local = _module
linear_model = _module
sem_ch_graph_conv = _module
sem_graph_conv = _module
progress = _module
bar = _module
counter = _module
helpers = _module
spinner = _module
viz = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


from torch import nn


import math


import torch.nn.functional as F


class _NonLocalBlock(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3,
        sub_sample=1, bn_layer=True):
        super(_NonLocalBlock, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
        assert self.inter_channels > 0
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        self.concat_project = nn.Sequential(nn.Conv2d(self.inter_channels *
            2, 1, 1, 1, 0, bias=False), nn.ReLU())
        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels,
                out_channels=self.in_channels, kernel_size=1, stride=1,
                padding=0), bn(self.in_channels))
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=
                self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample > 1:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample)
                )

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)
        phi_x = phi_x.expand(-1, -1, h, -1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class Linear(nn.Module):

    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        out = x + y
        return out


class LinearModel(nn.Module):

    def __init__(self, input_size, output_size, linear_size=1024, num_stage
        =2, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = input_size
        self.output_size = output_size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class SemCHGraphConv(nn.Module):
    """
    Semantic channel-wise graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemCHGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(2, in_features,
            out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj = adj.unsqueeze(0).repeat(out_features, 1, 1)
        self.m = self.adj > 0
        self.e = nn.Parameter(torch.zeros(out_features, len(self.m[0].
            nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.
                float))
            stdv = 1.0 / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0]).unsqueeze(1).transpose(1, 3)
        h1 = torch.matmul(input, self.W[1]).unsqueeze(1).transpose(1, 3)
        adj = -9000000000000000.0 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e.view(-1)
        adj = F.softmax(adj, dim=2)
        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E = E.unsqueeze(0).repeat(self.out_features, 1, 1)
        output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        output = output.transpose(1, 3).squeeze(1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(2, in_features,
            out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj = adj
        self.m = self.adj > 0
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=
            torch.float))
        nn.init.constant_(self.e.data, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.
                float))
            stdv = 1.0 / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        adj = -9000000000000000.0 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_garyzhao_SemGCN(_paritybench_base):
    pass
    def test_000(self):
        self._check(Linear(*[], **{'linear_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(LinearModel(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4])], {})

