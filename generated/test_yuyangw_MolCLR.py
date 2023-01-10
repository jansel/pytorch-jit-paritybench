import sys
_module = sys.modules[__name__]
del sys
dataset = _module
dataset_mix = _module
dataset_subgraph = _module
dataset_test = _module
finetune = _module
gcn_finetune = _module
gcn_molclr = _module
ginet_finetune = _module
ginet_molclr = _module
molclr = _module
nt_xent = _module

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


import time


import random


import numpy as np


from copy import deepcopy


import torch


import torch.nn.functional as F


from torch.utils.data.sampler import SubsetRandomSampler


import torchvision.transforms as transforms


import pandas as pd


from torch import nn


from torch.utils.tensorboard import SummaryWriter


from torch.optim.lr_scheduler import CosineAnnealingLR


from sklearn.metrics import roc_auc_score


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_absolute_error


from torch.nn import Parameter


from torch.nn import Linear


from torch.nn import LayerNorm


from torch.nn import ReLU


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


num_bond_direction = 3


num_bond_type = 5


num_atom_type = 119


num_chirality_tag = 3


class GCN(nn.Module):

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr='add'))
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        self.out_lin = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True), nn.Linear(self.feat_dim, self.feat_dim // 2))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        return h, out


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        self.out_lin = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True), nn.Linear(self.feat_dim, self.feat_dim // 2))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        return h, out


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        labels = torch.zeros(2 * self.batch_size).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NTXentLoss,
     lambda: ([], {'device': 0, 'batch_size': 4, 'temperature': 4, 'use_cosine_similarity': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_yuyangw_MolCLR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

