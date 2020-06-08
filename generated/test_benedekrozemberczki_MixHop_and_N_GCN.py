import sys
_module = sys.modules[__name__]
del sys
layers = _module
main = _module
param_parser = _module
trainer_and_networks = _module
utils = _module

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


import math


import torch


import random


class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.
            in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        feature_count, _ = torch.max(features['indices'], dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features['indices'], features['values'],
            feature_count[0], feature_count[1], self.weight_matrix)
        base_features = base_features + self.bias
        base_features = torch.nn.functional.dropout(base_features, p=self.
            dropout_rate, training=self.training)
        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix['indices'],
                normalized_adjacency_matrix['values'], base_features.shape[
                0], base_features.shape[0], base_features)
        return base_features


class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.
            in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features, p=self.
            dropout_rate, training=self.training)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix['indices'],
                normalized_adjacency_matrix['values'], base_features.shape[
                0], base_features.shape[0], base_features)
        base_features = base_features + self.bias
        return base_features


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Module initializing.
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


class NGCNNetwork(torch.nn.Module):
    """
    Higher Order Graph Convolutional Model.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, args, feature_number, class_number):
        super(NGCNNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.order = len(self.args.layers_1)
        self.setup_layer_structure()

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional layers) and dense final.
        """
        self.main_layers = [SparseNGCNLayer(self.feature_number, self.args.
            layers_1[i - 1], i, self.args.dropout) for i in range(1, self.
            order + 1)]
        self.main_layers = ListModule(*self.main_layers)
        self.fully_connected = torch.nn.Linear(sum(self.args.layers_1),
            self.class_number)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features = [self.main_layers[i](
            normalized_adjacency_matrix, features) for i in range(self.order)]
        abstract_features = torch.cat(abstract_features, dim=1)
        predictions = torch.nn.functional.log_softmax(self.fully_connected(
            abstract_features), dim=1)
        return predictions


class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args
            .layers_1[i - 1], i, self.args.dropout) for i in range(1, self.
            order_1 + 1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1,
            self.args.layers_2[i - 1], i, self.args.dropout) for i in range
            (1, self.order_2 + 1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.
            abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].
                weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].
                weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features_1 = torch.cat([self.upper_layers[i](
            normalized_adjacency_matrix, features) for i in range(self.
            order_1)], dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](
            normalized_adjacency_matrix, abstract_features_1) for i in
            range(self.order_2)], dim=1)
        predictions = torch.nn.functional.log_softmax(self.fully_connected(
            abstract_features_2), dim=1)
        return predictions


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_benedekrozemberczki_MixHop_and_N_GCN(_paritybench_base):
    pass
