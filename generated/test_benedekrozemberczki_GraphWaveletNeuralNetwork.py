import sys
_module = sys.modules[__name__]
del sys
gwnn = _module
gwnn_layer = _module
main = _module
param_parser = _module
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


from sklearn.model_selection import train_test_split


class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """

    def __init__(self, in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.ncount)], [node for node in range(self.ncount)]])
        self.diagonal_weight_indices = self.diagonal_weight_indices
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)


class DenseGraphWaveletLayer(GraphWaveletLayer):
    """
    Dense Graph Wavelet Layer Class.
    """

    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, features):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices, phi_values, self.diagonal_weight_indices, self.diagonal_weight_filter.view(-1), self.ncount, self.ncount, self.ncount)
        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices, rescaled_phi_values, phi_inverse_indices, phi_inverse_values, self.ncount, self.ncount, self.ncount)
        filtered_features = torch.mm(features, self.weight_matrix)
        localized_features = spmm(phi_product_indices, phi_product_values, self.ncount, self.ncount, filtered_features)
        return localized_features


class SparseGraphWaveletLayer(GraphWaveletLayer):
    """
    Sparse Graph Wavelet Layer Class.
    """

    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_indices, feature_values, dropout):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices, phi_values, self.diagonal_weight_indices, self.diagonal_weight_filter.view(-1), self.ncount, self.ncount, self.ncount)
        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices, rescaled_phi_values, phi_inverse_indices, phi_inverse_values, self.ncount, self.ncount, self.ncount)
        filtered_features = spmm(feature_indices, feature_values, self.ncount, self.in_channels, self.weight_matrix)
        localized_features = spmm(phi_product_indices, phi_product_values, self.ncount, self.ncount, filtered_features)
        dropout_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features), training=self.training, p=dropout)
        return dropout_features


class GraphWaveletNeuralNetwork(torch.nn.Module):
    """
    Graph Wavelet Neural Network class.
    For details see: Graph Wavelet Neural Network.
    Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng. ICLR, 2019
    :param args: Arguments object.
    :param ncount: Number of nodes.
    :param feature_number: Number of features.
    :param class_number: Number of classes.
    :param device: Device used for training.
    """

    def __init__(self, args, ncount, feature_number, class_number, device):
        super(GraphWaveletNeuralNetwork, self).__init__()
        self.args = args
        self.ncount = ncount
        self.feature_number = feature_number
        self.class_number = class_number
        self.device = device
        self.setup_layers()

    def setup_layers(self):
        """
        Setting up a sparse and a dense layer.
        """
        self.convolution_1 = SparseGraphWaveletLayer(self.feature_number, self.args.filters, self.ncount, self.device)
        self.convolution_2 = DenseGraphWaveletLayer(self.args.filters, self.class_number, self.ncount, self.device)

    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_indices, feature_values):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        deep_features_1 = self.convolution_1(phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_indices, feature_values, self.args.dropout)
        deep_features_2 = self.convolution_2(phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, deep_features_1)
        predictions = torch.nn.functional.log_softmax(deep_features_2, dim=1)
        return predictions

