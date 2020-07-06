import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
gat = _module
gat_layers = _module
gcn = _module
gcn_layers = _module
pscn = _module
train = _module
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


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data.sampler import Sampler


import sklearn


import itertools


import logging


from sklearn import preprocessing


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch.autograd import Variable


import time


import torch.optim as optim


from torch.utils.data import DataLoader


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import roc_auc_score


from sklearn.metrics import precision_recall_curve


class BatchMultiHeadGraphAttention(nn.Module):

    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(F.tanh(h_prime), self.a_src)
        attn_dst = torch.matmul(F.tanh(h_prime), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask, float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchGAT(nn.Module):

    def __init__(self, pretrained_emb, vertex_feature, use_vertex_feature, n_units=[1433, 8, 7], n_heads=[8, 1], dropout=0.1, attn_dropout=0.0, fine_tune=False, instance_normalization=False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)
        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)
        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=attn_dropout))

    def forward(self, x, vertices, adj):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj)
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)


class MultiHeadGraphAttention(nn.Module):

    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0)
        h_prime = torch.matmul(h.unsqueeze(0), self.w)
        attn_src = torch.bmm(h_prime, self.a_src)
        attn_dst = torch.bmm(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1)
        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(1 - adj, float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, lap):
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(lap, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchGCN(nn.Module):

    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature, use_vertex_feature, fine_tune=False, instance_normalization=False):
        super(BatchGCN, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)
        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)
        self.layer_stack = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer_stack.append(BatchGraphConvolution(n_units[i], n_units[i + 1]))

    def forward(self, x, vertices, lap):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)


class BatchPSCN(nn.Module):

    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature, use_vertex_feature, instance_normalization, neighbor_size, sequence_size, fine_tune=False):
        super(BatchPSCN, self).__init__()
        assert len(n_units) == 4
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)
        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)
        self.conv1 = nn.Conv1d(in_channels=n_units[0], out_channels=n_units[1], kernel_size=neighbor_size, stride=neighbor_size)
        k = 1
        self.conv2 = nn.Conv1d(in_channels=n_units[1], out_channels=n_units[2], kernel_size=k, stride=1)
        self.fc = nn.Linear(in_features=n_units[2] * (sequence_size - k + 1), out_features=n_units[3])

    def forward(self, x, vertices, recep):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        bs, l = recep.size()
        n = x.size()[1]
        offset = torch.ger(torch.arange(0, bs).long(), torch.ones(l).long() * n)
        offset = Variable(offset, requires_grad=False)
        offset = offset
        recep = (recep.long() + offset).view(-1)
        x = x.view(bs * n, -1)
        x = x.index_select(dim=0, index=recep)
        x = x.view(bs, l, -1)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(bs, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchGraphConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_xptree_DeepInf(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

