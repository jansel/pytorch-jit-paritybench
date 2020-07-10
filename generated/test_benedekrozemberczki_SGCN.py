import sys
_module = sys.modules[__name__]
del sys
main = _module
param_parser = _module
sgcn = _module
signedsageconvolution = _module
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


import time


import torch


import random


import numpy as np


import torch.nn.init as init


from torch.nn import Parameter


import torch.nn.functional as F


from sklearn.model_selection import train_test_split


import math


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class SignedSAGEConvolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """

    def __init__(self, in_channels, out_channels, norm=True, norm_embed=True, bias=True):
        super(SignedSAGEConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """

    def forward(self, x, edge_index):
        """
        Forward propagation pass with features an indices.
        :param x: Feature matrix.
        :param edge_index: Indices.
        """
        edge_index, _ = remove_self_loops(edge_index, None)
        row, col = edge_index
        if self.norm:
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        else:
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = torch.cat((out, x), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out


class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """

    def forward(self, x_1, x_2, edge_index_pos, edge_index_neg):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index_pos: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        edge_index_pos, _ = remove_self_loops(edge_index_pos, None)
        edge_index_pos, _ = add_self_loops(edge_index_pos, num_nodes=x_1.size(0))
        edge_index_neg, _ = remove_self_loops(edge_index_neg, None)
        edge_index_neg, _ = add_self_loops(edge_index_neg, num_nodes=x_2.size(0))
        row_pos, col_pos = edge_index_pos
        row_neg, col_neg = edge_index_neg
        if self.norm:
            out_1 = scatter_mean(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_mean(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))
        else:
            out_1 = scatter_add(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_add(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))
        out = torch.cat((out_1, out_2, x_1), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out


class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """

    def __init__(self, device, args, X):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1] * 2, self.neurons[0])
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1] * 2, self.neurons[0])
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1], self.neurons[i]))
            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1], self.neurons[i]))
        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        self.regression_weights = Parameter(torch.Tensor(4 * self.neurons[-1], 3))
        init.xavier_normal_(self.regression_weights)

    def calculate_regression_loss(self, z, target):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair.
        """
        pos = torch.cat((self.positive_z_i, self.positive_z_j), 1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j), 1)
        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k), 1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k), 1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k), 1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k), 1)
        features = torch.cat((pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        predictions = torch.mm(features, self.regression_weights)
        predictions_soft = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions_soft, target)
        return loss_term, predictions_soft

    def calculate_positive_embedding_loss(self, z, positive_edges):
        """
        Calculating the loss on the positive edge embedding distances
        :param z: Hidden vertex representation.
        :param positive_edges: Positive training edges.
        :return loss_term: Loss value on positive edge embedding.
        """
        self.positive_surrogates = [random.choice(self.nodes) for node in range(positive_edges.shape[1])]
        self.positive_surrogates = torch.from_numpy(np.array(self.positive_surrogates, dtype=np.int64).T)
        self.positive_surrogates = self.positive_surrogates.type(torch.long)
        positive_edges = torch.t(positive_edges)
        self.positive_z_i = z[(positive_edges[:, (0)]), :]
        self.positive_z_j = z[(positive_edges[:, (1)]), :]
        self.positive_z_k = z[(self.positive_surrogates), :]
        norm_i_j = torch.norm(self.positive_z_i - self.positive_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.positive_z_i - self.positive_z_k, 2, 1, True).pow(2)
        term = norm_i_j - norm_i_k
        term[term < 0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_negative_embedding_loss(self, z, negative_edges):
        """
        Calculating the loss on the negative edge embedding distances
        :param z: Hidden vertex representation.
        :param negative_edges: Negative training edges.
        :return loss_term: Loss value on negative edge embedding.
        """
        self.negative_surrogates = [random.choice(self.nodes) for node in range(negative_edges.shape[1])]
        self.negative_surrogates = torch.from_numpy(np.array(self.negative_surrogates, dtype=np.int64).T)
        self.negative_surrogates = self.negative_surrogates.type(torch.long)
        negative_edges = torch.t(negative_edges)
        self.negative_z_i = z[(negative_edges[:, (0)]), :]
        self.negative_z_j = z[(negative_edges[:, (1)]), :]
        self.negative_z_k = z[(self.negative_surrogates), :]
        norm_i_j = torch.norm(self.negative_z_i - self.negative_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.negative_z_i - self.negative_z_k, 2, 1, True).pow(2)
        term = norm_i_k - norm_i_j
        term[term < 0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_loss_function(self, z, positive_edges, negative_edges, target):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        loss_term_1 = self.calculate_positive_embedding_loss(z, positive_edges)
        loss_term_2 = self.calculate_negative_embedding_loss(z, negative_edges)
        regression_loss, self.predictions = self.calculate_regression_loss(z, target)
        loss_term = regression_loss + self.args.lamb * (loss_term_1 + loss_term_2)
        return loss_term

    def forward(self, positive_edges, negative_edges, target):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i - 1](self.h_pos[i - 1], self.h_neg[i - 1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i - 1](self.h_neg[i - 1], self.h_pos[i - 1], positive_edges, negative_edges)))
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        loss = self.calculate_loss_function(self.z, positive_edges, negative_edges, target)
        return loss, self.z

