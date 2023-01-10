import sys
_module = sys.modules[__name__]
del sys
dataset = _module
layers = _module
model = _module
train = _module
trainer = _module
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


import copy


import pandas as pd


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


import math


class DenseLayer(nn.Module):

    def __init__(self, config, weight_init, bias=False):
        super(DenseLayer, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        self.bn = config.dense_bn
        self.relu = config.dense_relu
        self.weight_init = weight_init
        self.dropout = nn.Dropout(config.drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)
        if config.accum == 'stack':
            self.bn_u = nn.BatchNorm1d(config.num_users * config.num_relations)
            self.bn_i = nn.BatchNorm1d((config.num_nodes - config.num_users) * config.num_relations)
        else:
            self.bn_u = nn.BatchNorm1d(config.num_users)
            self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users)
        self.relu = nn.ReLU()

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)
        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)
        return u_features, i_features


class BiDecoder(nn.Module):

    def __init__(self, config, weight_init):
        super(BiDecoder, self).__init__()
        self.num_basis = config.num_basis
        self.num_relations = config.num_relations
        self.feature_dim = config.hidden_size[1]
        self.accum = config.accum
        self.apply_drop = config.bidec_drop
        self.dropout = nn.Dropout(config.drop_prob)
        self.basis_matrix = nn.Parameter(torch.Tensor(config.num_basis, self.feature_dim * self.feature_dim))
        coefs = [nn.Parameter(torch.Tensor(config.num_basis)) for b in range(config.num_relations)]
        self.coefs = nn.ParameterList(coefs)
        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        nn.init.orthogonal_(self.basis_matrix)
        for coef in self.coefs:
            weight_init(coef, self.num_basis, self.num_relations)

    def forward(self, u_features, i_features):
        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)
        if self.accum == 'stack':
            u_features = u_features.reshape(self.num_relations, -1, self.feature_dim)
            i_features = i_features.reshape(self.num_relations, -1, self.feature_dim)
            num_users = u_features.shape[1]
            num_items = i_features.shape[1]
        else:
            num_users = u_features.shape[0]
            num_items = i_features.shape[0]
        for relation in range(self.num_relations):
            q_matrix = torch.sum(self.coefs[relation].unsqueeze(1) * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.feature_dim, self.feature_dim)
            if self.accum == 'stack':
                if relation == 0:
                    out = torch.chain_matmul(u_features[relation], q_matrix, i_features[relation].t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(u_features[relation], q_matrix, i_features[relation].t()).unsqueeze(-1)), dim=2)
            elif relation == 0:
                out = torch.chain_matmul(u_features, q_matrix, i_features.t()).unsqueeze(-1)
            else:
                out = torch.cat((out, torch.chain_matmul(u_features, q_matrix, i_features.t()).unsqueeze(-1)), dim=2)
        out = out.reshape(num_users * num_items, -1)
        return out


def split_stack(features, index, relations, dim_size):
    """
    Official Stack accumulation function

    Parameters
    ----------
    features : tensor (relation * num_nodes) x features
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    stacked_out : tensor(relation * nodes x out_dim)
    """
    out_dim = features.shape[0]
    np_index = index.numpy()
    np_relations = relations.numpy()
    splited_features = torch.split(features, int(out_dim / 5), dim=1)
    stacked_out = []
    for r, feature in enumerate(splited_features):
        relation_only_r = torch.from_numpy(np.where(np_relations == r)[0])
        r_index = index[relation_only_r]
        r_feature = feature[relation_only_r]
        stacked_out.append(scatter_('add', r_feature, r_index, dim_size=dim_size))
    stacked_out = torch.cat(stacked_out, 1)
    return stacked_out


def stack(features, index, relations, dim_size):
    """
    Stack accumulation function in RGCLayer.

    Parameters
    ----------
    features : tensor (relation * num_nodes)
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    out : tensor(relation * nodes x out_dim)
    """
    out = torch.zeros(dim_size * (torch.max(relations) + 1), features.shape[1])
    tar_idx = relations * dim_size + index
    out[tar_idx] = features
    return out


class GCEncoder(nn.Module):

    def __init__(self, config, weight_init):
        super(GCEncoder, self).__init__()
        self.num_relations = config.num_relations
        self.num_users = config.num_users
        self.accum = config.accum
        self.rgc_layer = RGCLayer(config, weight_init)
        self.dense_layer = DenseLayer(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm):
        features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)
        return u_features, i_features

    def separate_features(self, features):
        if self.accum == 'stack':
            num_nodes = int(features.shape[0] / self.num_relations)
            for r in range(self.num_relations):
                if r == 0:
                    u_features = features[:self.num_users]
                    i_features = features[self.num_users:(r + 1) * num_nodes]
                else:
                    u_features = torch.cat((u_features, features[r * num_nodes:r * num_nodes + self.num_users]), dim=0)
                    i_features = torch.cat((i_features, features[r * num_nodes + self.num_users:(r + 1) * num_nodes]), dim=0)
        else:
            u_features = features[:self.num_users]
            i_features = features[self.num_users:]
        return u_features, i_features


class GAE(nn.Module):

    def __init__(self, config, weight_init):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(config, weight_init)
        self.bidec = BiDecoder(config, weight_init)

    def forward(self, x, edge_index, edge_type, edge_norm):
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm)
        adj_matrices = self.bidec(u_features, i_features)
        return adj_matrices

