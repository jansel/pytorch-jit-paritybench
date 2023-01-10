import sys
_module = sys.modules[__name__]
del sys
aggregators = _module
dgn_layer = _module
scalers = _module
layers = _module
aggregators = _module
dgn_layer = _module
eigen_agg = _module
scalers = _module
HIV = _module
PCBA = _module
SBMs = _module
molecules = _module
multiplicity_eig = _module
superpixels = _module
main_HIV = _module
main_PCBA = _module
main_SBMs_node_classification = _module
main_molecules = _module
main_superpixels = _module
dgn_net = _module
dgn_net = _module
dgn_net = _module
aggregators = _module
dgn_layer = _module
layers = _module
mlp_readout_layer = _module
dgn_net = _module
dgn_net = _module
metrics = _module
train_COLLAB_edge_classification = _module
train_HIV_graph_classification = _module
train_PCBA_graph_classification = _module
train_SBMs_node_classification = _module
train_molecules_graph_regression = _module
train_superpixels_graph_classification = _module

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


from torch import nn


from functools import partial


import torch.nn as nn


import torch.nn.functional as F


import math


import scipy as sp


import time


from torch.utils.data import Dataset


import random as rd


from scipy import sparse as sp


import numpy as np


import itertools


import torch.utils.data


import scipy


import pandas as pd


from scipy.spatial.distance import cdist


import random


import torch.optim as optim


from torch.utils.data import DataLoader


from sklearn.metrics import confusion_matrix


from sklearn.metrics import f1_score


SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        return activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    """
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` with :math:`k=\\frac{1}{ \\text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0.0, b_norm=False, bias=True, init_fn=None, device='cpu'):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout, device=device)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_
        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_size) + ' -> ' + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='relu', last_activation='none', dropout=0.0, mid_b_norm=False, last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm, device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm, device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm, device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_size) + ' -> ' + str(self.out_size) + ')'


class DGNLayerComplex(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, residual, edge_features, edge_dim, pretrans_layers=1, posttrans_layers=1):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.residual = residual
        self.aggregators = aggregators
        self.scalers = scalers
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim, out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim, out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'], 'eig_d': edges.data['eig_d']}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        h_in = h
        g.ndata['h'] = h
        if self.edge_features:
            g.edata['ef'] = e
        g.apply_edges(self.pretrans_edges)
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)
        h = self.posttrans(h)
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNLayerSimple(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, residual, avg_d, posttrans_layers=1):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.aggregators = aggregators
        self.scalers = scalers
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.posttrans = MLP(in_size=len(aggregators) * len(scalers) * in_dim, hidden_size=out_dim, out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        return {'e': edges.src['h'], 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'], 'eig_d': edges.data['eig_d']}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        h_in = h
        g.ndata['h'] = h
        g.apply_edges(self.pretrans_edges)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        h = self.posttrans(h)
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNTower(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, pretrans_layers, posttrans_layers, edge_features, edge_dim):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.aggregators = aggregators
        self.scalers = scalers
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim, out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim, out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'], 'eig_d': edges.data['eig_d']}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        g.ndata['h'] = h
        if self.edge_features:
            g.edata['ef'] = e
        g.apply_edges(self.pretrans_edges)
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)
        h = self.posttrans(h)
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNLayerTower(nn.Module):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, batch_norm, towers=5, pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False, edge_dim=0):
        super().__init__()
        assert not divide_input or in_dim % towers == 0, 'if divide_input is set the number of towers has to divide in_dim'
        assert out_dim % towers == 0, 'the number of towers has to divide the out_dim'
        assert avg_d is not None
        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(DGNTower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators, scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout, graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim))
        self.mixing_network = FCLayer(out_dim, out_dim, activation='LeakyReLU')

    def forward(self, g, h, e, snorm_n):
        h_in = h
        if self.divide_input:
            h_cat = torch.cat([tower(g, h[:, n_tower * self.input_tower:(n_tower + 1) * self.input_tower], e, snorm_n) for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e, snorm_n) for tower in self.towers], dim=1)
        if len(self.towers) > 1:
            h_out = self.mixing_network(h_cat)
        else:
            h_out = h_cat
        if self.residual:
            h_out = h_in + h_out
        return h_out


EPS = 1e-05


def aggregate_dir_av(h, eig_s, eig_d, h_in, eig_idx):
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) / (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx(h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) / (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


def aggregate_dir_dx_balanced(h, eig_s, eig_d, h_in, eig_idx):
    eig_front = (torch.relu(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) / (torch.sum(torch.abs(torch.relu(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    eig_back = (torch.relu(eig_d[:, :, eig_idx] - eig_s[:, :, eig_idx]) / (torch.sum(torch.abs(-torch.relu(eig_d[:, :, eig_idx] - eig_s[:, :, eig_idx])), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    eig_w = (eig_front + eig_back) / 2
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


def aggregate_dir_dx_no_abs(h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) / (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in


def aggregate_dir_softmax(h, eig_s, eig_d, h_in, eig_idx, alpha):
    h_mod = torch.mul(h, torch.nn.Softmax(1)(alpha * torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]).unsqueeze(-1)))
    return torch.sum(h_mod, dim=1)


def aggregate_max(h, eig_s, eig_d, h_in):
    return torch.max(h, dim=1)[0]


def aggregate_mean(h, eig_s, eig_d, h_in):
    return torch.mean(h, dim=1)


def aggregate_min(h, eig_s, eig_d, h_in):
    return torch.min(h, dim=1)[0]


def aggregate_var(h, eig_s, eig_d, h_in):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_std(h, eig_s, eig_d, h_in):
    return torch.sqrt(aggregate_var(h, eig_s, eig_d, h_in) + EPS)


def aggregate_sum(h, eig_s, eig_d, h_in):
    return torch.sum(h, dim=1)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min, 'std': aggregate_std, 'var': aggregate_var, 'dir1-av': partial(aggregate_dir_av, eig_idx=1), 'dir2-av': partial(aggregate_dir_av, eig_idx=2), 'dir3-av': partial(aggregate_dir_av, eig_idx=3), 'dir1-0.1': partial(aggregate_dir_softmax, eig_idx=1, alpha=0.1), 'dir2-0.1': partial(aggregate_dir_softmax, eig_idx=2, alpha=0.1), 'dir3-0.1': partial(aggregate_dir_softmax, eig_idx=3, alpha=0.1), 'dir1-neg-0.1': partial(aggregate_dir_softmax, eig_idx=1, alpha=-0.1), 'dir2-neg-0.1': partial(aggregate_dir_softmax, eig_idx=2, alpha=-0.1), 'dir3-neg-0.1': partial(aggregate_dir_softmax, eig_idx=3, alpha=-0.1), 'dir1-dx': partial(aggregate_dir_dx, eig_idx=1), 'dir2-dx': partial(aggregate_dir_dx, eig_idx=2), 'dir3-dx': partial(aggregate_dir_dx, eig_idx=3), 'dir1-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=1), 'dir2-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=2), 'dir3-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=3), 'dir1-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=1), 'dir2-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=2), 'dir3-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=3)}


def scale_amplification(X, adj, avg_d=None):
    D = torch.sum(adj, -1)
    scale = (torch.log(D + 1) / avg_d['log']).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_attenuation(X, adj, avg_d=None):
    D = torch.sum(adj, -1)
    scale = (avg_d['log'] / torch.log(D + 1)).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_identity(X, adj, avg_d=None):
    return X


def scale_inverse_linear(X, adj, avg_d=None):
    D = torch.sum(adj, -1, keepdim=True)
    X_scaled = avg_d['lin'] * X / D
    return X_scaled


def scale_linear(X, adj, avg_d=None):
    D = torch.sum(adj, -1, keepdim=True)
    X_scaled = D * X / avg_d['lin']
    return X_scaled


SCALERS = {'identity': scale_identity, 'linear': scale_linear, 'inverse_linear': scale_inverse_linear, 'amplification': scale_amplification, 'attenuation': scale_attenuation}


class DGNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, type_net, residual, towers=5, divide_input=True, edge_features=None, edge_dim=None, pretrans_layers=1, posttrans_layers=1):
        super().__init__()
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]
        if type_net == 'simple':
            self.model = DGNLayerSimple(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm, batch_norm=batch_norm, residual=residual, aggregators=aggregators, scalers=scalers, avg_d=avg_d, posttrans_layers=posttrans_layers)
        elif type_net == 'complex':
            self.model = DGNLayerComplex(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm, batch_norm=batch_norm, aggregators=aggregators, residual=residual, scalers=scalers, avg_d=avg_d, edge_features=edge_features, edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers)
        elif type_net == 'towers':
            self.model = DGNLayerTower(in_dim=in_dim, out_dim=out_dim, aggregators=aggregators, scalers=scalers, avg_d=avg_d, dropout=dropout, graph_norm=graph_norm, batch_norm=batch_norm, towers=towers, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, divide_input=divide_input, residual=residual, edge_features=edge_features, edge_dim=edge_dim)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, decreasing_dim=True):
        super().__init__()
        if decreasing_dim:
            list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
            list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        else:
            list_FC_layers = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
            list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class DGNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        if self.edge_feat:
            self.embedding_e = nn.Linear(in_dim_edge, edge_dim)
        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm, batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat, edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _ in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout, graph_norm=self.graph_norm, batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat, edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.edge_feat:
            e = self.embedding_e(e)
        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            h = h_t
        g.ndata['h'] = h
        if self.readout == 'sum':
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')
        return self.MLP_layer(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class VirtualNode(nn.Module):

    def __init__(self, dim, dropout, batch_norm=False, bias=True, residual=True, vn_type='mean'):
        super().__init__()
        self.vn_type = vn_type.lower()
        self.fc_layer = FCLayer(in_size=dim, out_size=dim, activation='relu', dropout=dropout, b_norm=batch_norm, bias=bias)
        self.residual = residual

    def forward(self, g, h, vn_h):
        g.ndata['h'] = h
        if self.vn_type == 'mean':
            pool = mean_nodes(g, 'h')
        elif self.vn_type == 'sum':
            pool = sum_nodes(g, 'h')
        elif self.vn_type == 'logsum':
            pool = mean_nodes(g, 'h')
            lognum = torch.log(torch.tensor(g.batch_num_nodes, dtype=h.dtype, device=h.device))
            pool = pool * lognum.unsqueeze(-1)
        else:
            raise ValueError(f'Undefined input "{self.pooling}". Accepted values are "sum", "mean", "logsum"')
        vn_h_temp = self.fc_layer.forward(vn_h + pool)
        if self.residual:
            vn_h = vn_h + vn_h_temp
        else:
            vn_h = vn_h_temp
        temp_h = torch.cat([vn_h[ii:ii + 1].repeat(num_nodes, 1) for ii, num_nodes in enumerate(g.batch_num_nodes)], dim=0)
        h = h + temp_h
        return vn_h, h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FCLayer,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_size': 4, 'hidden_size': 4, 'out_size': 4, 'layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPReadout,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Saro00_DGN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

